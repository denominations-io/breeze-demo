package logit
package learning

import org.scalatest._

import org.apache.spark.sql._

import tooling._
import models._

class ModelEvaluationSpec extends FlatSpec with Matchers with DataReader with ModelEvaluation {

  val spark = SparkSession.builder().master("local[2]").appName("test-model-evaluation").getOrCreate()
  val (colNamesTrain, trainingData) = readCsv(spark, getClass.getResource("/wq-red-train.csv").getPath)
  val (colNamesHoldOut, holdOutData) = readCsv(spark, getClass.getResource("/wq-red-holdout.csv").getPath)

  "The model evaluation" should
    "evaluate the predictive performance of a trained model using holdout data" in {
      val predictions = new LogisticRegressionWithAdaGrad(spark, trainingData).predict(holdOutData)
      evaluate(spark, predictions, colNamesTrain)
      if(!spark.sparkContext.isStopped) spark.stop()
  }

  it should "be able to produce a summary report of diagnostic data on the model run" in {

  }

}
