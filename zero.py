# Simple Zero Rule algorithm implementation


def zero_rule_algorithm(train, test):
    output_values = [row[-1] for row in train]
    prediction = max(set(output_values), key=output_values.count)
    predicted = [prediction for i in range(len(test))]
    return predicted

if __name__ == "__main__":
    train = [['0'], ['0'], ['0'], ['0'], ['1'], ['1']]
    test = []
    for i in range(10):
        test.append([None])
    predictions = zero_rule_algorithm(train, test)

    print('Predictions: %s' % predictions)
