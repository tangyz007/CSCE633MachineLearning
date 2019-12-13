import csv


with open("satimage.scale.training") as fp:
    list=[]
    with open('parsed_training', mode='w') as training:
        training_writer = csv.writer(training)
        for x in range(0,37):
            if x==0:
                feature="Class"
            else:
                feature ="feature"+str(x)
            list.append(feature)
        training_writer.writerow(list)



    line = fp.readline()
    while line:
        data = line.split()
        parsedDataList=[]
        for x in range(0,37):
            parsedDataList.append("")

        for item in data:
            index=item.find(":")
            if index>0:
                featureNumber=item[0:index]
                featureData=item[index+1:]
                parsedDataList[int(featureNumber)]=featureData
                # print('number is ',featureNumber,' data is ',item[index+1:])
            else:
                parsedDataList[0]=item
        # print(parsedDataList)
        with open('parsed_training.csv', mode='a') as training:
            training_writer = csv.writer(training)
            training_writer.writerow(parsedDataList)
        line = fp.readline()


    # print(data)
    # while line:
    #     data = line.split()
        # print(line)
