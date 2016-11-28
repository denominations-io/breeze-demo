package logit
package models

import logit.learning.ModelEvaluation
import org.scalatest._

import org.apache.spark.sql.SparkSession

import tooling._

class MllibLogisticRegressionWithSGDSpec extends FlatSpec with Matchers with DataReader with ModelEvaluation {

  val spark = SparkSession.builder().master("local[2]").appName("test-mllib-sgd").getOrCreate()

  "The MllibLogisticRegressionWithSGD" should
    "estimate parameter coefficients for the logistic function using MLLib's ml.classification implementation of SGD" in {
      val (colNames, trainingData) = readCsv(spark, getClass.getResource("/wq-red-train.csv").getPath)
      val (colNamesDup, holdOutData) = readCsv(spark, getClass.getResource("/wq-red-holdout.csv").getPath)

      val model = new MLLibLogisticRegressionWithSGD(trainingData)
      val predictions = model.predict(holdOutData)
      evaluate(spark, predictions, colNames)
      if(!spark.sparkContext.isStopped) spark.stop()
  }
}
