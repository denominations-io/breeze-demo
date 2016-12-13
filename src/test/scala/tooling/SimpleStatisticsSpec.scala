import org.apache.spark.sql.SparkSession
import org.scalatest._

class SimpleStatisticsSpec extends FlatSpec with Matchers with DataReader {

  val spark = SparkSession.builder().master("local[2]").appName("test-simple-statistics").getOrCreate()
  val (colNames, data) =  readCsv(spark, getClass.getResource("/train.csv").getPath)

  "The SimpleStatistics" should
    "summarize a tabular data set with some simple statistics" in {
      val summary = SimpleStatistics(colNames, data)
      summary.contains("number of observations") shouldEqual true
      summary.contains("column means") shouldEqual true
      summary.contains("column variances") shouldEqual true
  }
}
