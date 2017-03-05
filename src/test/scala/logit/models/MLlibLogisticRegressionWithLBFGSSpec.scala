import org.apache.spark.sql._
import org.scalatest._

class MLlibLogisticRegressionWithLBFGSSpec extends FlatSpec with Matchers with DataReader {

  val spark = SparkSession.builder().master("local[2]").appName("test-mllib-lbfgs").getOrCreate()
  val (colNames, trainingData) = readCsv(spark, getClass.getResource("/train.csv").getPath)
  val (colNamesDup, holdOutData) = readCsv(spark, getClass.getResource("/holdout.csv").getPath)

  "The MLlibLogisticRegressionWithLBFGS" should
    "be able to estimate parameter coefficients correctly" in {
    val model = new MLlibLogisticRegressionWithLBFGS(spark, colNames, trainingData, holdOutData)
    val f = model.evaluate
    f.generateSummary("target/model-evaluation-specs/mllib_lbfgs.txt")
  }

}
