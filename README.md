## Breeze demo app

demo app - machine learning in Scala with the Spark MLLib and ScalaNLP Breeze libraries. two different versions of a logistic regression are implemented using MLLib and Breeze.

the wine quality data is part of the the UCI Machine Learning Repository, courtesy of Cortez et al. 2009. *Modeling wine preferences by data mining from physicochemical properties.* Decision Support Systems, 47(4):547-553.

## Dependencies

* JVM
* SBT
* Spark 2.0.2

## Running the app

Compile the project with `sbt assembly` and run the jar with the following parameters:

```
breeze-demo-assembly-0.1.jar \
    --training <training_data>.csv \
    --holdout <holdout_set>.csv \
    --optimizer adagrad
```
