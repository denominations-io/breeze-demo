import org.scalatest._
import org.apache.spark.sql._
import breeze.linalg._
import breeze.optimize._
import breeze.stats.distributions._

class LogisticRegressionWithAdaGradSpec extends FlatSpec with Matchers with DataReader {

  val spark = SparkSession.builder().master("local[2]").appName("test-breeze-adagrad").getOrCreate()
  val (colNames, trainingData) = readCsv(spark, getClass.getResource("/train.csv").getPath)
  val (colNamesDup, holdOutData) = readCsv(spark, getClass.getResource("/holdout.csv").getPath)

  "The LogisticRegressionWithAdaGrad" should
   "be able to estimate parameter coefficients using breeze.optimize.AdaptiveGradientDescent" in {
      val model = new LogisticRegressionWithAdaGrad(spark, colNames, trainingData, holdOutData)
      model.evaluate.evaluationSet shouldEqual 2101
      // TODO: add more assertions; cross-validate the estimates.

  }

  it should "produce a correct gradient" in {
    val model = new LogisticRegressionWithAdaGrad(spark, colNames, trainingData, holdOutData)
    GradientTester
      .test[Int, DenseVector[Double]](model.regression, DenseVector.rand[Double](colNames.length, RandBasis.mt0.uniform)) // TODO: look into NaNs
  }
}
