import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

under_score = ['goal0.0'] #Anything to be replaced with '_' put in this list.
for f in os.listdir("."):
    copy_f = f
    for char in copy_f:
        if (char in under_score): copy_f = copy_f.replace(char,'goal5.0')
    os.rename(f,copy_f)
