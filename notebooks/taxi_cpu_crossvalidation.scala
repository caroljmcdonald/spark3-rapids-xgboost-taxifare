// Databricks notebook source
// MAGIC %md # Taxi CrossValidation with GPU accelerating on XGBoost
// MAGIC 
// MAGIC In this notebook, we will show you how to levarage GPU to accelerate taxi CrossValidation on XGBoost to find out the best model given a group parameters.
// MAGIC 
// MAGIC ## Import classes
// MAGIC First we need load some common classes that both GPU version and CPU version will use:

// COMMAND ----------


import ml.dmlc.xgboost4j.scala.spark.{XGBoostRegressionModel, XGBoostRegressor}

import org.apache.spark._
import org.apache.spark.ml._
import org.apache.spark.ml.feature._
import org.apache.spark.ml.regression._
import org.apache.spark.ml.evaluation._
import org.apache.spark.ml.tuning._
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.ml.Pipeline

// COMMAND ----------

// MAGIC %md what is new to xgboost-spark users is rapids.GpuDataReader and **rapids.CrossValidator**

// COMMAND ----------




// COMMAND ----------

// MAGIC %md ## Set dataset path

// COMMAND ----------


//val trainPath= "/FileStore/tables/taxismall.csv"

// You need to update them to your real paths!
//"/FileStore/tables/taxi_tsmall.csv"
//"/FileStore/tables/taxi_esmall.csv"
val trainPath= "dbfs:/taxi_train_parquet" 
val evalPath= "dbfs:/taxi_eval_parquet" 

// COMMAND ----------

// MAGIC %md # Set the schema of the dataset

// COMMAND ----------


val labelColName = "fare_amount"

/* schema is not needed for parquet
val schema =
    StructType(Array(
      StructField("vendor_id", FloatType),
      StructField("passenger_count", FloatType),
      StructField("trip_distance", FloatType),
      StructField("pickup_longitude", FloatType),
      StructField("pickup_latitude", FloatType),
      StructField("rate_code", FloatType),
      StructField("store_and_fwd", FloatType),
      StructField("dropoff_longitude", FloatType),
      StructField("dropoff_latitude", FloatType),
      StructField("fare_amount", FloatType),
      StructField("trip_time",LongType),
      StructField("year", IntegerType),
      StructField("month", IntegerType),
      StructField("day", FloatType),
      StructField("day_of_week", FloatType),
      StructField("is_weekend", FloatType),
      StructField("hour", FloatType),
      StructField("h_distance",DoubleType)
    ))
 */   

// COMMAND ----------

// MAGIC %md ## Create a new spark session and load data
// MAGIC we must create a new spark session to continue all spark operations. It will also be used to initilize the `GpuDataReader` which is a data reader powered by GPU.
// MAGIC 
// MAGIC NOTE: in this notebook, we have uploaded dependency jars when installing toree kernel. If we don't upload them at installation time, we can also upload in notebook by [%AddJar magic](https://toree.incubator.apache.org/docs/current/user/faq/). However, there's one restriction for `%AddJar`: the jar uploaded can only be available when `AddJar` is called after a new spark session is created. We must use it as below:
// MAGIC 
// MAGIC ```scala
// MAGIC import org.apache.spark.sql.SparkSession
// MAGIC val spark = SparkSession.builder().appName("Taxi-GPU").getOrCreate
// MAGIC %AddJar file:/data/libs/cudf-0.14-SNAPSHOT-cuda10.jar
// MAGIC %AddJar file:/data/libs/xgboost4j_3.0-1.0.0-SNAPSHOT.jar
// MAGIC %AddJar file:/data/libs/xgboost4j-spark_3.0-1.0.0-SNAPSHOT.jar
// MAGIC %AddJar file:/data/libs/rapids-4-spark-0.1-SNAPSHOT.jar
// MAGIC // ...
// MAGIC ```

// COMMAND ----------

val sparkSession = SparkSession.builder().appName("taxi-GPU").getOrCreate

val reader =  sparkSession.read
var tdf= reader.parquet(trainPath)
// small csv
//var df = reader.option("inferSchema", "false").option("header", true).schema(schema).csv(path)
// val Array(tdf, edf) = df.randomSplit(Array(0.8, 0.2), seed = 1234L)


// COMMAND ----------

tdf.schema

// COMMAND ----------

tdf.cache
tdf.createOrReplaceTempView("taxi")
spark.catalog.cacheTable("taxi")

// COMMAND ----------

tdf.head

// COMMAND ----------

// MAGIC %python
// MAGIC import numpy as np
// MAGIC import pandas as pd
// MAGIC pdf = spark.table("taxi")
// MAGIC padf = pd.DataFrame(pdf.take(3),  columns=pdf.columns)
// MAGIC padf.head()
// MAGIC 
// MAGIC 
// MAGIC 
// MAGIC   
// MAGIC   

// COMMAND ----------

// MAGIC %python
// MAGIC import numpy as np
// MAGIC import pandas as pd
// MAGIC pdf = spark.table("taxi")
// MAGIC pdf.describe().toPandas().transpose()
// MAGIC   
// MAGIC   

// COMMAND ----------

display(tdf.select("passenger_count", "h_distance","fare_amount","trip_time", "pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude").describe())

// COMMAND ----------

tdf.select("passenger_count", "h_distance","fare_amount","trip_time").describe().show

// COMMAND ----------

display(tdf.select("fare_amount"))

// COMMAND ----------

display(tdf.select("h_distance"))

// COMMAND ----------

display(tdf.select("trip_time"))

// COMMAND ----------

  var cor = tdf.stat.corr("fare_amount", "h_distance")

// COMMAND ----------

cor = tdf.stat.corr("fare_amount", "trip_time")

// COMMAND ----------

tdf.select(corr("fare_amount","pickup_longitude")).show()

// COMMAND ----------

tdf.select(corr("fare_amount","hour")).show()

// COMMAND ----------

tdf.select(corr("fare_amount","day_of_week")).show()

// COMMAND ----------

tdf.select(corr("fare_amount","rate_code")).show()

// COMMAND ----------

// MAGIC %python
// MAGIC from pyspark.sql.functions import col, rand
// MAGIC 
// MAGIC for col in tdf.columns:
// MAGIC   correlation = tdf.stat.corr("fare_amount", col)
// MAGIC   
// MAGIC   print("The correlation between columns 'fare_amount' and '{}': \t{}".format(col, correlation))

// COMMAND ----------

// MAGIC %python
// MAGIC from pyspark.ml.feature import VectorAssembler
// MAGIC from pyspark.ml.stat import Correlation
// MAGIC import numpy as np
// MAGIC import pandas as pd
// MAGIC import matplotlib.pyplot as plt
// MAGIC import seaborn as sns
// MAGIC try:
// MAGIC   tdf
// MAGIC except NameError: # Looks for local table if not defined
// MAGIC   tdf= spark.table("taxi")
// MAGIC 
// MAGIC assembler = VectorAssembler(inputCols=tdf.columns, outputCol="features")
// MAGIC tdfv = assembler.transform(tdf)
// MAGIC 
// MAGIC pearsonCorr = Correlation.corr(tdfv, 'features').collect()[0][0]
// MAGIC pandasDF = pd.DataFrame(pearsonCorr.toArray())
// MAGIC 
// MAGIC pandasDF.index, pandasDF.columns = tdf.columns, tdf.columns
// MAGIC 
// MAGIC fig, ax = plt.subplots()
// MAGIC sns.heatmap(pandasDF)
// MAGIC display(fig.figure)

// COMMAND ----------

tdf.groupBy("hour").count.orderBy($"hour").show(24)

// COMMAND ----------

tdf.groupBy("hour").count.orderBy($"hour").explain

// COMMAND ----------

// MAGIC %sql
// MAGIC select hour, avg(fare_amount)
// MAGIC from taxi
// MAGIC group by hour order by hour 

// COMMAND ----------

// MAGIC %sql
// MAGIC select pickup_latitude, pickup_longitude, avg(fare_amount)
// MAGIC from taxi
// MAGIC group by pickup_latitude, pickup_longitude 

// COMMAND ----------

// MAGIC %sql
// MAGIC select dropoff_latitude, dropoff_longitude, avg(fare_amount)
// MAGIC from taxi
// MAGIC group by dropoff_latitude, dropoff_longitude 

// COMMAND ----------



// COMMAND ----------

// MAGIC %sql
// MAGIC select month, avg(fare_amount)
// MAGIC from taxi
// MAGIC group by month order by month

// COMMAND ----------

// MAGIC %sql
// MAGIC select day_of_week, avg(fare_amount)
// MAGIC from taxi
// MAGIC group by day_of_week order by day_of_week

// COMMAND ----------



// COMMAND ----------

// MAGIC %sql
// MAGIC select trip_distance,  avg(fare_amount)
// MAGIC from taxi
// MAGIC group by trip_distance order by  avg(fare_amount) desc

// COMMAND ----------

// MAGIC %sql
// MAGIC select *
// MAGIC from taxi

// COMMAND ----------

// MAGIC %sql
// MAGIC select trip_time, fare_amount
// MAGIC from taxi

// COMMAND ----------

// MAGIC %sql
// MAGIC select *
// MAGIC from taxi

// COMMAND ----------

display(tdf.select("fare_amount"))

// COMMAND ----------

display(tdf.select("h_distance"))

// COMMAND ----------

// MAGIC %md ## Features to train

// COMMAND ----------


var featureNames = Array("h_distance", "pickup_longitude","pickup_latitude","dropoff_longitude", "dropoff_latitude","rate_code","hour", "day_of_week","trip_time")

// COMMAND ----------

val assembler = new VectorAssembler().setInputCols(featureNames).setOutputCol("features")
val trdf = assembler.transform(tdf)

// COMMAND ----------

val regressorParam = Map(
    "learning_rate" -> 0.05,
    "gamma" -> 1,
    "objective" ->"reg:gamma",
    "max_depth" -> 8,
    "subsample" -> 0.8,
    "num_round" -> 100,
    "tree_method" -> "hist")

// COMMAND ----------

// MAGIC %md ## Construct CrossValidator

// COMMAND ----------


//import ml.dmlc.xgboost4j.scala.spark.rapids.CrossValidator
val regressor = new XGBoostRegressor(regressorParam)
    .setLabelCol(labelColName)

val paramGrid = new ParamGridBuilder()
    .addGrid(regressor.maxDepth, Array(3, 8))
    .addGrid(regressor.eta, Array(0.2, 0.6))
    .build()
val evaluator = new RegressionEvaluator().setLabelCol(labelColName)
val cv = new CrossValidator()
    .setEstimator(regressor)
    .setEvaluator(evaluator)
    .setEstimatorParamMaps(paramGrid)
    .setNumFolds(3)
    .setParallelism(4)

// COMMAND ----------

regressor.explainParams()

// COMMAND ----------

// MAGIC %md ## train with CrossValidator

// COMMAND ----------

val cvmodel = cv.fit(trdf)
val model =cvmodel.bestModel.asInstanceOf[XGBoostRegressionModel]

// COMMAND ----------

cvmodel.getEstimatorParamMaps.zip(cvmodel.avgMetrics)

// COMMAND ----------

model.extractParamMap()

// COMMAND ----------

// MAGIC %sh
// MAGIC ls -l /dbfs/taximodel

// COMMAND ----------


val path = "/dbfs/taximodel/taxixgbrmodel"
model.write.overwrite().save(path)


// COMMAND ----------

val nativeModelPath = "/dbfs/taximodel/xgboost_native_model"
model.nativeBooster.saveModel(nativeModelPath)


// COMMAND ----------

var edf= reader.parquet(evalPath)

// COMMAND ----------

val erdf = assembler.transform(edf)

// COMMAND ----------

// MAGIC %md ## tranform with best model trained by CrossValidator

// COMMAND ----------

val pdf = model.transform(erdf).cache()
pdf.select("fare_amount", "prediction").describe().show()

// COMMAND ----------


val evaluator = new RegressionEvaluator().setLabelCol(labelColName)
val rmse = evaluator.evaluate(pdf)
println(s"RMSE is $rmse")

// COMMAND ----------

val r2 = evaluator.setMetricName("r2").evaluate(pdf)

println(s"R2 is $r2")

// COMMAND ----------

val maevaluator =evaluator.setMetricName("mae").evaluate(pdf)

// COMMAND ----------

var predictions = pdf.withColumn("error",  $"fare_amount" - $"prediction" )

// COMMAND ----------

val avgFare = tdf.select(avg("fare_amount")).first().getDouble(0)
predictions = predictions.withColumn("avgPrediction", lit(avgFare))
val regressionMeanEvaluator = new RegressionEvaluator()
  .setPredictionCol("avgPrediction")
  .setLabelCol("fare_amount")
  .setMetricName("rmse")

val rmsep = regressionMeanEvaluator.evaluate(predictions)


// COMMAND ----------

predictions.select("fare_amount", "prediction", "avgPrediction", "error").describe().show()

// COMMAND ----------

val df=predictions
df.cache
df.createOrReplaceTempView("taxi")
spark.catalog.cacheTable("taxi")

// COMMAND ----------


spark.close()

// COMMAND ----------


