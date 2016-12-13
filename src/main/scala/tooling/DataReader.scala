import org.apache.spark.sql._

import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg._

trait DataReader {
  def readCsv(spark: SparkSession, location: String): (Array[String], Dataset[LabeledPoint]) = {

    val rawData = spark.read.format("CSV").option("header", "true").option("delimiter", ",").load(location)

    import spark.implicits._

    val colNames = rawData.schema.map { _.name }
    val features = colNames.slice(0, colNames.size - 1).toArray

    val data = rawData.map { case row: Row =>
        val features: Vector =
          Vectors.dense(
            row
              .toSeq
              .map { dp => try dp.toString.toDouble catch { case e: NumberFormatException => 0.0 } }
              .toArray
              .slice(0, row.size - 1)
          )
        LabeledPoint(row.getString(row.size - 1).toDouble, features)
      }
    (features, data)
  }

}
