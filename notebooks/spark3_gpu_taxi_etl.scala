// Databricks notebook source
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.FloatType


// COMMAND ----------

#%sh
# uncomment to download csv data 
#!mkdir -p taxi_data
#!cd taxi_data 
# download 6 months or one month, note the schema changed after june 2016
#wget https://s3.amazonaws.com/nyc-tlc/trip+data/yellow_tripdata_2016-{01..6}.csv
#wget https://s3.amazonaws.com/nyc-tlc/trip+data/yellow_tripdata_2016-02.csv
#took about an hour (51.77 minutes) 63 GB total

// COMMAND ----------


lazy val schema =
StructType(Array(
StructField("VendorId", IntegerType),
StructField("tpep_pickup_datetime", TimestampType),
StructField("tpep_dropoff_datetime", TimestampType),
StructField("passenger_count", IntegerType),
StructField("trip_distance", FloatType),
StructField("pickup_longitude", FloatType),
StructField("pickup_latitude", FloatType),
StructField("RatecodeID", IntegerType),
StructField("store_and_fwd_flag", StringType),
StructField("dropoff_longitude", FloatType),
StructField("dropoff_latitude", FloatType),
StructField("payment_type", IntegerType),
StructField("fare_amount", FloatType),
StructField("extra", FloatType),
StructField("mta_tax", FloatType),
StructField("tip_amount", FloatType),
StructField("tolls_amount", FloatType),
StructField("improvement_surcharge", FloatType),
StructField("total_amount", FloatType) 
  ))

// read multiple files or one file yellow_tripdata_2016-02.csv
var df = spark.read.option("inferSchema", "false").option("header", true).schema(schema).csv("dbfs:/taxi_data/*.csv")  




// COMMAND ----------

//write the data to parquet format
df.write.parquet("dbfs:/taxi_data_parquet")

// COMMAND ----------

var df = spark.read.parquet("dbfs:/taxi_data_parquet")



// COMMAND ----------

df.createOrReplaceTempView("taxi")

// COMMAND ----------

// MAGIC %python
// MAGIC import numpy as np
// MAGIC import pandas as pd
// MAGIC pdf = spark.table("taxi")
// MAGIC pdf.describe().toPandas().transpose()

// COMMAND ----------

def dropUseless(dataFrame: DataFrame): DataFrame = {
    dataFrame.drop(
      "payment_type",
      "extra",
      "mta_tax",
      "tip_amount",
      "tolls_amount",
      "improvement_surcharge",
      "total_amount")
  }

// COMMAND ----------

df = dropUseless(df)

// COMMAND ----------

// rename columne
df = df.withColumnRenamed("tpep_pickup_datetime", "pickup_datetime")
df = df.withColumnRenamed("tpep_dropoff_datetime", "dropoff_datetime")
df = df.withColumnRenamed("VendorID", "vendor_id")
df = df.withColumnRenamed("store_and_fwd_flag", "store_and_fwd")
df = df.withColumnRenamed("RatecodeID", "rate_code")

// COMMAND ----------

// drop null values 
df = df.na.drop()


// COMMAND ----------

/* It is also possible to combine this into one statement
val df2=df.withColumn("trip_time",unix_timestamp($"dropoff_time")-unix_timestamp($"pickup_time"))
.withColumn("pickup_hour",hour($"pickup_time"))
.withColumn("pickup_day",dayofweek($"pickup_time"))
.withColumn("pickup_day_month",dayofmonth($"pickup_time"))
.withColumn("pickup_minute",minute($"pickup_time"))
.withColumn("pickup_weekday",weekofyear($"pickup_time"))
.withColumn("pickup_month",month($"pickup_time"))
.filter($"fare">0 || $"trip_time">0 || $"fare"<5000)
.drop("dropoff_time","pickup_time")
*/

// COMMAND ----------

// filter out anomalous values
df = df.filter(!($"trip_distance" < 1 and $"fare_amount" > 15 ))
df = df.filter(!($"trip_distance" < 10 and $"fare_amount" > 40 ))
df = df.filter($"trip_distance" > 0 and $"trip_distance" < 100)
df = df.filter($"fare_amount" > 0 and $"fare_amount" < 100)
df = df.filter($"passenger_count" > 0 and $"passenger_count" <= 6)
df = df.filter($"pickup_longitude" > -75 and $"pickup_longitude" < -72)
df = df.filter($"dropoff_longitude" > -75 and $"dropoff_longitude" < -72)
df = df.filter($"pickup_latitude" > 40 and $"pickup_latitude" < 42)
df = df.filter($"dropoff_latitude" > 40 and $"dropoff_latitude" < 42)


// COMMAND ----------

// change Y,N to 1,0
df = df.withColumn("store_and_fwd",
    when(col("store_and_fwd") === "Y", 1).otherwise(0))

// COMMAND ----------

// calcuate trip_time
val start = unix_timestamp(col("pickup_datetime")).cast(LongType)
val end = unix_timestamp(col("dropoff_datetime")).cast(LongType)
df = df.withColumn("trip_time", (end-start))
//df = df.withColumn("trip_time",unix_timestamp($"dropoff_time")-unix_timestamp($"pickup_time"))


// COMMAND ----------

// filter out anomouslous trip_time values
df = df.filter(!($"trip_time" < 1000 and $"fare_amount" > 40 ))
df = df.filter($"trip_time" > 10 )
df = df.filter($"trip_time" <  40000)

// COMMAND ----------

// get day of week and hour
val datetime = col("pickup_datetime")
df = df.withColumn("year", year(datetime)).withColumn("month", month(datetime))
      .withColumn("day", dayofmonth(datetime))
      .withColumn("day_of_week", dayofweek(datetime))
      .withColumn("is_weekend",col("day_of_week").isin(1, 7).cast(IntegerType))  // 1: Sunday, 7: Saturday
      .withColumn("hour", hour(datetime))




// COMMAND ----------

// drop columns no longer needed
df= df.drop( "pickup_datetime")
df= df.drop( "dropoff_datetime")


// COMMAND ----------

// calculate haversine distance  
def addHDistance(dataFrame: DataFrame): DataFrame = {
    val P = math.Pi / 180
    val lat1 = col("pickup_latitude")
    val lon1 = col("pickup_longitude")
    val lat2 = col("dropoff_latitude")
    val lon2 = col("dropoff_longitude")
    val internalValue = (lit(0.5)
      - cos((lat2 - lat1) * P) / 2
      + cos(lat1 * P) * cos(lat2 * P) * (lit(1) - cos((lon2 - lon1) * P)) / 2)
    val hDistance = lit(12734) * asin(sqrt(internalValue))
    dataFrame.withColumn("h_distance", hDistance)
  }

// COMMAND ----------

df = addHDistance(df)

// COMMAND ----------

// filter out anomalous distance values
df = df.filter(!($"h_distance" < 1 and $"fare_amount" >15))
df = df.filter(!($"h_distance" < 10 and $"fare_amount" > 40 ))
df = df.filter($"h_distance" > 1 and $"h_distance" < 100)


// COMMAND ----------

// round off location values
df = df.withColumn("pickup_longitude", round( df("pickup_longitude"),3)).withColumn("pickup_latitude", round( df("pickup_latitude"),3))
df = df.withColumn("dropoff_longitude", round( df("dropoff_longitude"),3)).withColumn("dropoff_latitude", round( df("dropoff_latitude"),3))

// COMMAND ----------

df = df.withColumn("h_distance", round( $"h_distance",3))

// COMMAND ----------

df.schema

// COMMAND ----------

#%sh
#rm -r /dbfs/taxi_etl_parquet

// COMMAND ----------

df.write.mode("overwrite").parquet("dbfs:/taxi_etl_parquet")

// COMMAND ----------

df.unpersist()

// COMMAND ----------

var df = spark.read.parquet("dbfs:/taxi_etl_parquet")

// COMMAND ----------

df.cache
df.createOrReplaceTempView("taxi")
spark.catalog.cacheTable("taxi")

// COMMAND ----------

// MAGIC %python
// MAGIC import numpy as np
// MAGIC import pandas as pd
// MAGIC pdf = spark.table("taxi")
// MAGIC pdf.describe().toPandas().transpose()

// COMMAND ----------

val Array(train, eval, test, rest) = df.randomSplit(Array(0.5, 0.15, 0.15, 0.20), seed = 1234L)

// COMMAND ----------


train.cache
train.select("passenger_count", "trip_distance","h_distance","rate_code","fare_amount").describe().show

// COMMAND ----------

// MAGIC %sh
// MAGIC rm -r /dbfs/taxi_train_parquet
// MAGIC rm -r /dbfs/taxi_eval_parquet
// MAGIC rm -r /dbfs/taxi_test_parquet

// COMMAND ----------

train.write.parquet("dbfs:/taxi_train_parquet")

// COMMAND ----------

train.unpersist

// COMMAND ----------

eval.write.parquet("dbfs:/taxi_eval_parquet")

// COMMAND ----------

eval.unpersist

// COMMAND ----------

test.write.parquet("dbfs:/taxi_test_parquet")

// COMMAND ----------

rest.coalesce(1).write.format("com.databricks.spark.csv").option("header", "true").save("dbfs:/FileStore/carolmtaxicsv")



// COMMAND ----------

rest.schema

// COMMAND ----------

// MAGIC %sh
// MAGIC ls -l /dbfs/FileStore/carolmtaxicsv/taxismall.csv

// COMMAND ----------

df.unpersist
test.unpersist

// COMMAND ----------

rest.unpersist

// COMMAND ----------

var df = spark.read.parquet("dbfs:/taxi_train_parquet")

// COMMAND ----------


df.cache
df.createOrReplaceTempView("taxi")
spark.catalog.cacheTable("taxi")

// COMMAND ----------

display(df.groupBy("hour").count.orderBy("hour"))

// COMMAND ----------

display(df.select($"hour", $"fare_amount"))

// COMMAND ----------

df.select("passenger_count", "trip_distance","h_distance","rate_code","fare_amount").describe().show

// COMMAND ----------

// MAGIC %sql
// MAGIC select trip_distance , h_distance from taxi

// COMMAND ----------

// MAGIC %sql
// MAGIC select trip_distance, rate_code, fare_amount, is_weekend, day_of_week, h_distance from taxi

// COMMAND ----------

 df.select("trip_distance", "h_distance","fare_amount").show(5)

// COMMAND ----------

// MAGIC %sql
// MAGIC select hour, avg(fare_amount)
// MAGIC from taxi
// MAGIC group by hour order by hour 

// COMMAND ----------

// MAGIC %sql
// MAGIC select hour, avg(fare_amount), avg(trip_distance)
// MAGIC from taxi
// MAGIC group by hour order by hour 

// COMMAND ----------

// MAGIC %sql
// MAGIC select hour, avg(fare_amount),  avg(trip_time)
// MAGIC from taxi
// MAGIC group by hour order by hour 

// COMMAND ----------

// MAGIC %sql
// MAGIC select rate_code, avg(fare_amount) , avg(trip_distance)
// MAGIC from taxi
// MAGIC group by rate_code order by rate_code

// COMMAND ----------

// MAGIC %sql
// MAGIC select h_distance, fare_amount
// MAGIC from taxi
