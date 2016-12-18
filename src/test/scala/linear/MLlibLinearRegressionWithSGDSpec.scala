import org.scalatest._
import org.apache.spark.sql._

class MLlibLinearRegressionWithSGDSpec extends FlatSpec with Matchers with DataReader {

  val spark = SparkSession.builder().master("local[2]").appName("test-mllib-linear-sgd").getOrCreate()
  val (colNames, trainingData) = readCsv(spark, getClass.getResource("/train.csv").getPath)
  val (colNamesDup, holdOutData) = readCsv(spark, getClass.getResource("/holdout.csv").getPath)

  "The MLLibLinearRegressionWithSGD" should
    "be able to estimate and evaluate a linear regression" in {
    val model = new MLlibLinearRegressionWithSGD(colNames, trainingData, holdOutData)
    model.evaluate.logSummary()
  }

}
