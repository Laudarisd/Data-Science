# Predicting Which Passenger Survived the Titanic

This project is probably the rite of passage for everyone getting into data science. I never really enjoyed the movie, but the door could totally fit both Rose and Jack.

This project is binary classification problem, where the passenger either survived (`1`) or died (`0`). Here is a list of the columns of the dataset:

* `PassengerID` - Unique ID for each column
* `Survived` - Whether the passenger survived (`1`) or not (`0`)
* `Pclass` - Class of the passenger's ticket. Either 1, 2 or 3.
* `Sex` - Passenger's sex (male or female)
* `Age` - Passenger's age
* `Sibsp` - Number of sibling or spouses aboard the Titanic
* `Parch` - Number of parents or children aboard the Titanic
* `Ticket` - Passenger's ticket number
* `Fare` - The price paid for the passenger's ticket
* `Cabin` - Passenger's cabin number
* `Embarked` - Port where the passenger embarked. Can be:
    * `C` - Cherbourg
    * `Q` - Queenstown
    * `S` - Southampton

Although we know exactly who survived the Titanic, the project is still useful to apply important concepts in data science and machine learning. So here it is!

**Objective:** Predict which passenger survived the Titanic (Jack died)

**Techniques used:**
* Pandas, matplotlib, numpy
* Scikit-learn
* Logistic regression, cross-validation, k-nearest neighbours
* Regular expressions
* Heatmap
* Recursive feature elimination
* Hyperparameter optimization
* Grid search
* Random forest classifier
