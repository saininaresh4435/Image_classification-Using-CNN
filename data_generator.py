def data_generator(input_loc):
    import os,cv2
    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn.utils import shuffle
    from sklearn.cross_validation import train_test_split

    from keras import backend as K
    K.set_image_dim_ordering('th')

    from keras.utils import np_utils
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation, Flatten
    from keras.layers.convolutional import Convolution2D, MaxPooling2D
    from keras.optimizers import SGD,RMSprop,adam
    
    img_rows=128
    img_cols=128
    num_channel=1
    num_epoch=10

    num_classes = 2
    img_data_list=[]
    data_dir_list=os.listdir(input_loc)
    
    
    USE_SKLEARN_PREPROCESSING=True
    if USE_SKLEARN_PREPROCESSING:
        # using sklearn for preprocessing
        from sklearn import preprocessing

        def image_to_feature_vector(image, size=(128, 128)):
            # resize the image to a fixed size, then flatten the image into
            # a list of raw pixel intensities
            return cv2.resize(image, size).flatten()

        img_data_list=[]
        for dataset in data_dir_list:
            img_list=os.listdir(input_loc+'/'+ dataset)
            for img in img_list:
                input_img=cv2.imread(input_loc + '/'+ dataset + '/'+ img )
                input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
                input_img_flatten=image_to_feature_vector(input_img,(128,128))
                img_data_list.append(input_img_flatten)

        img_data = np.array(img_data_list)
        img_data = img_data.astype('float32')
        img_data_scaled = preprocessing.scale(img_data)

        if K.image_dim_ordering()=='th':
            img_data_scaled=img_data_scaled.reshape(img_data.shape[0],num_channel,img_rows,img_cols)
            print (img_data_scaled.shape)

        else:
            img_data_scaled=img_data_scaled.reshape(img_data.shape[0],img_rows,img_cols,num_channel)
            print (img_data_scaled.shape)
            
    if USE_SKLEARN_PREPROCESSING:
        img_data=img_data_scaled
    num_classes = 2

    num_of_samples = img_data.shape[0]
    labels = np.ones((num_of_samples,),dtype='int64')

    labels[0:100]=0
    labels[100:]=1

    names = ['Driver_No_Phone', 'Driver_Phone']

    # convert class labels to on-hot encoding
    Y = np_utils.to_categorical(labels, num_classes)

    #Shuffle the dataset
    x,y = shuffle(img_data,Y, random_state=2)
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
    input_shape=img_data[0].shape
    
    model = Sequential()

    model.add(Convolution2D(32, 3,3,border_mode='same',input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=["accuracy"])
    #model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=["accuracy"])

    # Viewing model_configuration

    model.summary()
    
    
    hist = model.fit(X_train, y_train, batch_size=16, nb_epoch=num_epoch, verbose=1, validation_data=(X_test, y_test))
    
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test Loss:', score[0])
    print('Test accuracy:', score[1])
    
    from sklearn.metrics import classification_report,confusion_matrix
    import itertools

    Y_pred = model.predict(X_test)
    print("Y_pred\n")
    print(Y_pred)
    y_pred = np.argmax(Y_pred, axis=1)
    print(y_pred)
    y_pred = model.predict_classes(X_test)
    print(y_pred)
    target_names = ['class 0(N)', 'class 1(Y)']

    print("\n")
    print("Confusion Matrix\n")
    print(confusion_matrix(np.argmax(y_test,axis=1), y_pred))