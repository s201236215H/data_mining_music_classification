# data_mining_music_classification

This is a project that classify 4 genre of music(jazz,pop,classical,rock), The repository consist of a classification program, the GTZAN dataset and pyAudioAnalysis library.

### Program description
The program first extract 4 features named ZCR, energyEntropy, SpectralEntropy, Spectral Flux, 13-D MFCC. Then, we get the mean-feature-vector from each music. After that, We train the classifer by 50 tracks from each genre. Finally, we yield the result by predicting labels for other 25 songs.

###Run the code
1. Download python,pip and the repository (Try not to run python on windows, may got some solvable problems)
2. Goto `genre` directory
3. ```pip install -r requirement.txt```
4. ```python m_classify.m```

###Unfinished task
The task of data mining should use cross-validation set instead of only training and testing set. We can use the library of sklearn for more accuracy.  
####more info: 
http://scikit-learn.org/stable/modules/cross_validation.html
