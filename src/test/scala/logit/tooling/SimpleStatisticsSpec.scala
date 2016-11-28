package logit
package tooling

import org.scalatest._

import org.apache.spark.sql.SparkSession

class SimpleStatisticsSpec extends FlatSpec with Matchers with DataReader {

  val spark = SparkSession.builder().master("local[2]").appName("test-simple-statistics").getOrCreate()
  val (colNames, data) =  readCsv(spark, getClass.getResource("/wq-red-train.csv").getPath)

  "The SimpleStatistics" should
    "summarize a tabular data set with some simple statistics" in {
      val summary = SimpleStatistics(colNames, data)
      summary.contains("number of observations") shouldEqual true
      summary.contains("column means") shouldEqual true
      summary.contains("column variances") shouldEqual true
      if(!spark.sparkContext.isStopped) spark.stop()
  }
}
