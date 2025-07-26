import csv
import sys
import datetime

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    data = (list(), list())

    # Store each evidence, label in the tuple "data".
    with open("shopping.csv", "r") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            header = row
            break
        for row in reader:
            data[0].append(list(row[:-1]))
            data[1].append(row[-1])

    # Columns of the type "float".
    float_types = [
        'Administrative_Duration', 'Informational_Duration', 'ProductRelated_Duration', 
        'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay'
    ]

    # Columns of the type "int".
    int_types = [
        'Administrative', 'Informational', 'ProductRelated', 'OperatingSystems', 
        'Browser', 'Region', 'TrafficType'
    ]

    # Convert the evidences.
    for i in range(len(data[0])):
        for j in range(len(data[0][i])):
            
            if header[j] in float_types:
                data[0][i][j] = float(data[0][i][j])
            
            if header[j] in int_types:
                data[0][i][j] = int(data[0][i][j])
            
            if header[j] == header[-2]:
                data[0][i][j] = 0 if data[0][i][j] == "FALSE" else 1
            
            if header[j] == header[-3]:
                data[0][i][j] = 0 if data[0][i][j] == 'New_Visitor' else 1
            
            if header[j] == 'Month':
                try:
                    data[0][i][j] = datetime.datetime.strptime(data[0][i][j], "%b").month - 1
                except ValueError:
                    data[0][i][j] = datetime.datetime.strptime(data[0][i][j], "%B").month - 1

    # Convert the labels.
    for i in range(len(data[1])):
        data[1][i] = 0 if data[1][i] == 'FALSE' else 1

    return data


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """

    model = KNeighborsClassifier()

    model.fit(evidence, labels)

    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    real_positives = real_negatives = true_positives = true_negatives = 0
    for i in range(len(labels)):
        if labels[i] == 0:
            real_negatives += 1
            if labels[i] == predictions[i]:
                true_negatives += 1
        else:
            real_positives += 1
            if labels[i] == predictions[i]:
                true_positives += 1

    return (true_positives / real_positives, true_negatives / real_negatives)


if __name__ == "__main__":
    main()
    