# Autopan-Python
Autopan code written in python

# How it works
We calculate the moving average to lower the sensitivy of new values. This moving average tells us where to move. Since we are doing a moving average, a lag is introduced. To compinsate for the lag, we lag the production of the video too so that then the timing gets synced back up. We also mask everything outside the court so that the model only considers the players that are on the court. 
