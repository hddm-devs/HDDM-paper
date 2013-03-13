running first attempt for subjects experiment
first set of jobs was run using commit: 51d6075b034f5d22ddbec0e4e31843d7e0de42dc

in one data set the non-decision time was very low and min(RT) was 0.009, which is smaller then initial value set for t by the model (0.01).
so I reduce the initial value of t and st to 0.001
I also had to change EstimationSingleOptimization and GroupOptimization to use my_hddm.HDDMGamma, becuase till now they were set to use hddm.HDDMTruncated 
so I could not set their initial value without doing changes to the hddm module.
I run again (only a few simulations left) using commit: 633d32ebc6a590d08c60588e6734f938ce6fe8cb
