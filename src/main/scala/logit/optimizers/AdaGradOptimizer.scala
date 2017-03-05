package logit
package optimizers

import breeze.linalg._
import breeze.optimize._

import breeze.util._

trait AdaGradOptimizer extends SerializableLogging {

  val regValue = 2.3
  val batchSize = 0.01
  val maxIterations = 200
  val tolerance: Double = 1E-8

  def optimizeAGD(f: StochasticDiffFunction[DenseVector[Double]], parameters: DenseVector[Double]): DenseVector[Double] = {
    val adaGrad =
      new AdaptiveGradientDescent.L2Regularization[DenseVector[Double]](regularizationConstant = regValue,
                                                                        stepSize = batchSize,
                                                                        maxIter = maxIterations,
                                                                        tolerance = tolerance)

    val state = adaGrad.minimizeAndReturnState(f, parameters)
    logger.info("adjusted gradient: " + state.adjustedGradient)
    logger.info("adjusted value: " + state.adjustedValue)
    state.x
  }

}
