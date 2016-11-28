package logit
package tooling

import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession
import org.scalatest._

class DataReaderSpec extends FlatSpec with Matchers with DataReader {

  val spark = SparkSession.builder().master("local[2]").appName("test-datareader").getOrCreate()

  "The DataReader" should
    "map the data in a .csv file to a data set of ml LabeledPoints" in {
       val (colNames, trainingData) = readCsv(spark, getClass.getResource("/wq-red-train.csv").getPath)

       colNames.size shouldEqual 11
       trainingData.count() shouldEqual 1000
       trainingData.collect().take(1)(0) shouldEqual
         LabeledPoint(0.0, Vectors.dense(Array(7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4)))

       if(!spark.sparkContext.isStopped) spark.stop()
    }
}
