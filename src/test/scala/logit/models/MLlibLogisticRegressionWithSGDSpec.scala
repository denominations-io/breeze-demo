import org.scalatest._
import org.apache.spark.sql.SparkSession

class MLlibLogisticRegressionWithSGDSpec extends FlatSpec with Matchers with DataReader with ModelEvaluation {

  val spark = SparkSession.builder().master("local[2]").appName("test-mllib-sgd").getOrCreate()

  val (colNames, trainingData) = readCsv(spark, getClass.getResource("/train.csv").getPath)
  val (colNamesDup, holdOutData) = readCsv(spark, getClass.getResource("/holdout.csv").getPath)

  val model = new MLlibLogisticRegressionWithSGD(colNames, trainingData, holdOutData)

  "The MLLib logistic regression" should
    "estimate parameter coefficients for the logistic function using MLLib's ml.classification implementation of SGD" in {
      model.parameterEstimates
  }

  it should "correctly estimate a gradient function from the data for stochastic gradient descent" in {
    model.logisticGradient
  }

}
