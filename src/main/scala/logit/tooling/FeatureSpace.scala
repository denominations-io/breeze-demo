package logit
package tooling

import breeze.linalg._
import org.apache.spark.ml.feature.LabeledPoint

trait FeatureSpace {
  def featureMatrixFromLabeledPoint(features: Int, data: Array[LabeledPoint]): DenseMatrix[Double] = {
    val featureMatrix = DenseMatrix.zeros[Double](data.length, features)
    val observations = data.map { lp => DenseVector(lp.features.toArray).t }

    val rowIndices = 0 to data.size
    (rowIndices, observations).zipped.foreach { (i, row) => featureMatrix(i, ::) := row }
    featureMatrix
  }
}
