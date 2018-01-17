def getInputs():
    myfile = open('data/LSTMDataVentriclesForecastLeaderBoardTrainParsed.csv','r')
    i = 0
    inputs,targets = [],[]
    for line in myfile:
        if i == 0:
            i += 1
            continue
        arr = line.split(',')
        inputs.append(list(map(float, arr[1:-1])))
        targets.append([float(arr[-1])])
    return inputs,targets