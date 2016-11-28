package logit
package optimizers

import breeze.linalg._
import breeze.optimize._

trait AdaGradOptimizer {

  val regValue = 3
  val batchSize = 25
  val maxIterations = 200
  val tolerance: Double = 1E-4

  def optimizeAGD(f: StochasticDiffFunction[DenseVector[Double]], parameters: DenseVector[Double]): DenseVector[Double] = {
    val adaGrad = new AdaptiveGradientDescent.L2Regularization[DenseVector[Double]](
      regularizationConstant = regValue, stepSize = batchSize, maxIter = maxIterations, tolerance = tolerance)

    adaGrad.minimize(f, parameters)
  }

}
