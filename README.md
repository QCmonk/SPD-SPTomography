# SPD-SPTomography
Demonstration code for performing state and process tomography using matlab CVX and the matlab engine in python. 

This code will not work without some modifcation as it was designed to be part of a much larger library. It requires matlab to be installed as well as CVX for matlab. If you wish to use my optimisation methods rather than write your own you will need to modify tomography.py to accept your measurement record as well as modify the preperation and state tomography rotations to match those used in your experiment. I used what I believe to be the most standard basis however you really ought to check. There are about a thousand other caveats however since this repository is meant as nothing more than an example I'll leave it here. 

You are welcome to contact me at josh.morris@monash.edu if you require assistance/advice/help in getting state/process tomography running using either an SDP approach or some other estimation method. 
