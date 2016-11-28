package logit
package tooling

import breeze.util.SerializableLogging
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.sql.Dataset

object SimpleStatistics extends SerializableLogging {

  def apply(featureSet: Array[String], data: Dataset[LabeledPoint]): String = {

    def format(vector: org.apache.spark.mllib.linalg.Vector): String =
      vector.toArray
        .zip(featureSet)
        .map(colVar => s"\t${colVar._2}: ${colVar._1}")
        .mkString("\n")

    val observations = data.count()

    val stats = Statistics
      .colStats(data.rdd.map { lp => Vectors.dense(lp.features.toArray) })
    val columnMeans = format(stats.mean)
    val columnVariances = format(stats.variance)

    val summary =
      Array(
        s"number of observations: $observations",
        s"column means: \n $columnMeans",
        s"column variances: \n $columnVariances"
      ).mkString("\n")
    logger.info(summary)
    summary
  }
}
