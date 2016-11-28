package logit
package models

import org.scalatest._

import org.apache.spark.sql._

import breeze.linalg._
import breeze.optimize._
import breeze.stats.distributions._

import tooling._

class LogisticRegressionWithAdaGradSpec extends FlatSpec with Matchers with DataReader {

  val spark = SparkSession.builder().master("local[2]").appName("test-breeze-adagrad").getOrCreate()
  val (colNamesTrain, trainingData) = readCsv(spark, getClass.getResource("/wq-red-train.csv").getPath)

  "The LogisticRegressionWithAdaGrad" should
   "be able to estimate parameter coefficients using breeze.optimize.AdaptiveGradientDescent" in {
      val model = new LogisticRegressionWithAdaGrad(spark, trainingData)
      model.estimate.size shouldEqual colNamesTrain.length
      // TODO: add more assertions; cross-validate the estimates.

  }

  it should "produce a correct gradient" in {
    val model = new LogisticRegressionWithAdaGrad(spark, trainingData)
    // val coefficients = model.optimizedParameters
    val results = GradientTester.test[Int, DenseVector[Double]](model.regression, DenseVector.rand[Double](colNamesTrain.length, RandBasis.mt0.uniform))
    if(!spark.sparkContext.isStopped) spark.stop()
  }
}
