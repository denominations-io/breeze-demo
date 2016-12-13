import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession
import org.scalatest._

class DataReaderSpec extends FlatSpec with Matchers with DataReader {

  val spark = SparkSession.builder().master("local[2]").appName("test-datareader").getOrCreate()

  "The DataReader" should
    "map the data in a .csv file to a data set of ml LabeledPoints" in {
       val (colNames, trainingData) = readCsv(spark, getClass.getResource("/train.csv").getPath)

       colNames.size shouldEqual 57
       trainingData.count() shouldEqual 2500
       trainingData.collect().take(1)(0) shouldEqual
         LabeledPoint(1,
           Vectors
             .dense(
               Array(
                 0.17,
                 0.17,
                 0.69,
                 0,
                 0.34,
                 0.17,
                 0,
                 0.86,
                 0.17,
                 0.69,
                 0.34,
                 1.38,
                 0,
                 0,
                 0,
                 0,
                 1.73,
                 0.34,
                 2.07,
                 1.55,
                 3.8,
                 0,
                 0,
                 0.34,
                 0,
                 0,
                 0,
                 0,
                 0,
                 0,
                 0,
                 0,
                 0,
                 0,
                 0,
                 0,
                 0,
                 0,
                 0,
                 0.17,
                 0,
                 0,
                 0,
                 0,
                 0,
                 0,
                 0,
                 0,
                 0,
                 0.194,
                 0,
                 1.718,
                 0.055,
                 0,
                 5.175,
                 63,
                 621
               )
             )
         )

    }
}
