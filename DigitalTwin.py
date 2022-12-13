from telegram.ext  import *
import pandas as pd
import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import substring, col
import dataframe_image as dfi
def twin():
    spark = SparkSession.builder.master("local").appName("wind_turbine_project").getOrCreate()
    sc = spark.sparkContext
    spark_df = spark.read.csv('T1.csv', header=True, inferSchema=True)
    spark_df.cache()
    spark_df = spark_df.toDF(*[c.lower() for c in spark_df.columns])
    spark_df = spark_df.withColumn("month", substring("date/time", 4,2))
    spark_df = spark_df.withColumn("hour", substring("date/time", 12,2))
    spark_df = spark_df.withColumn("day", substring("date/time", 12,2))
    spark_df = spark_df.withColumn('month', spark_df.month.cast(IntegerType()))
    spark_df = spark_df.withColumn('hour', spark_df.hour.cast(IntegerType()))
    spark_df = spark_df.withColumn('day', spark_df.month.cast(IntegerType()))
    pd.options.display.float_format = '{:.2f}'.format
    spark_df.select('wind speed (m/s)', 'theoretical_power_curve (kwh)', 'lv activepower (kw)').toPandas().describe()
    sample_df = spark_df.sample(withReplacement=False, fraction=0.1, seed=42).toPandas()
    columns = ['wind speed (m/s)', 'wind direction (°)', 'theoretical_power_curve (kwh)', 'lv activepower (kw)']
    wind_speed = spark_df.select('wind speed (m/s)').toPandas()
    spark_df = spark_df.withColumn('activepower (kw)', spark_df['lv activepower (kw)'])
    variables = ['month', 'hour', 'wind speed (m/s)', 'wind direction (°)']
    vectorAssembler = VectorAssembler(inputCols = variables, outputCol = 'features')
    va_df = vectorAssembler.transform(spark_df)
    final_df = va_df.select('features', 'activepower (kw)')
    splits = final_df.randomSplit([0.8, 0.2])
    train_df = splits[0]
    test_df = splits[1]
    train_c= 'Training dataset : '+str(train_df.count())
    test_c= 'Testing dataset : '+str(test_df.count())
    gbm = GBTRegressor(featuresCol='features', labelCol='activepower (kw)')
    gbm_model = gbm.fit(train_df)
    y_pred = gbm_model.transform(test_df)
    evaluator = RegressionEvaluator(predictionCol='prediction', labelCol='activepower (kw)')
    r2=evaluator.evaluate(y_pred, {evaluator.metricName: 'r2'})
    mae=evaluator.evaluate(y_pred, {evaluator.metricName: 'mae'})
    rmse=evaluator.evaluate(y_pred, {evaluator.metricName: 'rmse'})
    eva_df = spark.createDataFrame(sample_df)
    eva_df = eva_df.withColumn('activepower (kw)', eva_df['lv activepower (kw)'])
    variables = ['month', 'hour', 'wind speed (m/s)', 'wind direction (°)']
    vectorAssembler = VectorAssembler(inputCols = variables, outputCol = 'features')
    vec_df = vectorAssembler.transform(eva_df)
    vec_df = vec_df.select('features', 'activepower (kw)')
    preds = gbm_model.transform(vec_df)
    preds_df = preds.select('activepower (kw)','prediction').toPandas()
    frames = [sample_df[['wind speed (m/s)', 'theoretical_power_curve (kwh)']], preds_df]
    sample_data = pd.concat(frames, axis=1)
    return sample_data,train_c,test_c,r2,mae,rmse

print('Twin Running.....')
def start_command(update,context):
    update.message.reply_text("Twin Started.....")

def handle_msg(update,context):
    text=str(update.message.text)
    response,train_c,test_c,r2,mae,rmse = twin()
    dfi.export(response.sample(int(text)) , 'result.png')
    update.message.reply_text(train_c)
    update.message.reply_text(test_c)
    update.message.reply_text('Showing '+text+' Random rows:')
    update.message.reply_photo(photo=open('result.png', 'rb'))
    update.message.reply_text('R² Score : '+str(r2))
    update.message.reply_text('MAE : '+str(mae))
    update.message.reply_text('RMSE : '+str(rmse))

if __name__ == '__main__':
    updater = Updater('5418702082:AAE4HThDbgpYUR3m0hKaBggcXZ6Qo_RbhyY',use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("Start",start_command))
    dp.add_handler(MessageHandler(Filters.text,handle_msg))
    updater.start_polling()
    updater.idle()
