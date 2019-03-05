def bonus_knn(self,train_data,train_label,i):
    from sklearn.model_selection import train_test_split  
    X_train, X_test, y_train, y_test = train_test_split(train_data,train_label, test_size=0.20,shuffle=False)
    print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
    from sklearn.preprocessing import StandardScaler  
    scaler = StandardScaler()  
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)  
    X_test = scaler.transform(X_test)

    from sklearn.neighbors import KNeighborsClassifier  
    classifier = KNeighborsClassifier(n_neighbors=5)  
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    
    acc = {}

    # Calculating error for K values between 1 and 40
    for i in range(2, 6):  
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, y_train)
        pred_i = knn.predict(X_test)
        acc[i]=(np.mean(pred_i == y_test))
    print(i,acc)
    return acc