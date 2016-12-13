import org.scalatest._
import org.apache.spark.sql._

class MLLibLinearRegressionWithSGDSpec extends FlatSpec with Matchers with DataReader {

  val spark = SparkSession.builder().master("local[2]").appName("test-mllib-linear-sgd").getOrCreate()
  val (colNamesTrain, trainingData) = readCsv(spark, getClass.getResource("/train.csv").getPath)
  val (colNamesDup, holdOutData) = readCsv(spark, getClass.getResource("/holdout.csv").getPath)

  "The MLLibLinearRegressionWithSGD" should
    "be able to estimate and evaluate a linear regression" in {
    val model = new MLLibLinearRegressionWithSGD(trainingData)
    model.evaluation(holdOutData).logSummary()
  }

}
