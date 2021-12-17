import pandas as pd
import matplotlib.pyplot as plt


path='.\\full_full_v2.csv'
df=pd.read_csv(path)



random9 = df[df.trainset == "('random', 0.9)"]
random8 = df[df.trainset == "('random', 0.8)"]
random7 = df[df.trainset == "('random', 0.7)"]
random6 = df[df.trainset == "('random', 0.6)"]
row2 = df[df.trainset == "('rows', 2)"]
row4 = df[df.trainset == "('rows', 4)"]
row6 = df[df.trainset == "('rows', 6)"]
col2 = df[df.trainset == "('cols', 2)"]
col4 = df[df.trainset == "('cols', 4)"]
col6 = df[df.trainset == "('cols', 6)"]




# frame = random9

rat_mean = pd.DataFrame()
rat_std = pd.DataFrame()
ep_mean = pd.DataFrame()
ep_std = pd.DataFrame()
fac_mean = pd.DataFrame()
fac_std = pd.DataFrame()

for frame in [random9,random8,random7,random6]:
    print(frame[frame.average_error == frame.average_error.min()])

    rat_mean = rat_mean.append(frame[["trainset","min_ratings",'average_error']].groupby(["trainset","min_ratings"]).mean())
    rat_std = rat_std.append(frame[["trainset","min_ratings",'average_error']].groupby(["trainset","min_ratings"]).std())

    ep_mean = ep_mean.append(frame[["trainset","n_epochs",'average_error']].groupby(["trainset","n_epochs"]).mean())
    ep_std = ep_std.append(frame[["trainset","n_epochs",'average_error']].groupby(["trainset","n_epochs"]).std())

    fac_mean = fac_mean.append(frame[["trainset","n_factors",'average_error']].groupby(["trainset","n_factors"]).mean())
    fac_std = fac_std.append(frame[["trainset","n_factors",'average_error']].groupby(["trainset","n_factors"]).std())

rat_mean = rat_mean.reset_index().pivot(index = 'min_ratings', columns = 'trainset')
rat_std = rat_std.reset_index().pivot(index = 'min_ratings', columns = 'trainset')

ep_mean = ep_mean.reset_index().pivot(index = 'n_epochs', columns = 'trainset')
ep_std = ep_std.reset_index().pivot(index = 'n_epochs', columns = 'trainset')

fac_mean = fac_mean.reset_index().pivot(index = 'n_factors', columns = 'trainset')
fac_std = fac_std.reset_index().pivot(index = 'n_factors', columns = 'trainset')

# breakpoint()

plt1 = rat_mean.plot.bar(yerr=rat_std)
plt2 = ep_mean.plot.bar(yerr=ep_std)
plt3 = fac_mean.plot.bar(yerr=fac_std)

for i, p in enumerate([plt1,plt2,plt3]):
    p.set_ylabel("Average Error")
    p.legend(["(random, 0.6)", "(random, 0.7)", "(random, 0.8)", "(random, 0.9)"], loc = "lower left")
    plt.savefig('..\\..\\pics for report\\rand'+str(i)+".jpg")



rat_mean = pd.DataFrame()
rat_std = pd.DataFrame()
ep_mean = pd.DataFrame()
ep_std = pd.DataFrame()
fac_mean = pd.DataFrame()
fac_std = pd.DataFrame()

for frame in [row2,row4, row6]:
    print(frame[frame.average_error == frame.average_error.min()])

    rat_mean = rat_mean.append(frame[["trainset","min_ratings",'average_error']].groupby(["trainset","min_ratings"]).mean())
    rat_std = rat_std.append(frame[["trainset","min_ratings",'average_error']].groupby(["trainset","min_ratings"]).std())

    ep_mean = ep_mean.append(frame[["trainset","n_epochs",'average_error']].groupby(["trainset","n_epochs"]).mean())
    ep_std = ep_std.append(frame[["trainset","n_epochs",'average_error']].groupby(["trainset","n_epochs"]).std())

    fac_mean = fac_mean.append(frame[["trainset","n_factors",'average_error']].groupby(["trainset","n_factors"]).mean())
    fac_std = fac_std.append(frame[["trainset","n_factors",'average_error']].groupby(["trainset","n_factors"]).std())

rat_mean = rat_mean.reset_index().pivot(index = 'min_ratings', columns = 'trainset')
rat_std = rat_std.reset_index().pivot(index = 'min_ratings', columns = 'trainset')

ep_mean = ep_mean.reset_index().pivot(index = 'n_epochs', columns = 'trainset')
ep_std = ep_std.reset_index().pivot(index = 'n_epochs', columns = 'trainset')

fac_mean = fac_mean.reset_index().pivot(index = 'n_factors', columns = 'trainset')
fac_std = fac_std.reset_index().pivot(index = 'n_factors', columns = 'trainset')

plt1 = rat_mean.plot.bar(yerr=rat_std)
plt2 = ep_mean.plot.bar(yerr=ep_std)
plt3 = fac_mean.plot.bar(yerr=fac_std)

for i, p in enumerate([plt1,plt2,plt3]):
    p.set_ylabel("Average Error")
    p.legend(["(rows, 2)", "(rows, 4)", "(rows, 6)"], loc = "lower left")
    plt.savefig('..\\..\\pics for report\\rows'+str(i)+".jpg")



rat_mean = pd.DataFrame()
rat_std = pd.DataFrame()
ep_mean = pd.DataFrame()
ep_std = pd.DataFrame()
fac_mean = pd.DataFrame()
fac_std = pd.DataFrame()

for frame in [col2, col4, col6]:
    print(frame[frame.average_error == frame.average_error.min()])

    rat_mean = rat_mean.append(frame[["trainset","min_ratings",'average_error']].groupby(["trainset","min_ratings"]).mean())
    rat_std = rat_std.append(frame[["trainset","min_ratings",'average_error']].groupby(["trainset","min_ratings"]).std())

    ep_mean = ep_mean.append(frame[["trainset","n_epochs",'average_error']].groupby(["trainset","n_epochs"]).mean())
    ep_std = ep_std.append(frame[["trainset","n_epochs",'average_error']].groupby(["trainset","n_epochs"]).std())

    fac_mean = fac_mean.append(frame[["trainset","n_factors",'average_error']].groupby(["trainset","n_factors"]).mean())
    fac_std = fac_std.append(frame[["trainset","n_factors",'average_error']].groupby(["trainset","n_factors"]).std())

rat_mean = rat_mean.reset_index().pivot(index = 'min_ratings', columns = 'trainset')
rat_std = rat_std.reset_index().pivot(index = 'min_ratings', columns = 'trainset')

ep_mean = ep_mean.reset_index().pivot(index = 'n_epochs', columns = 'trainset')
ep_std = ep_std.reset_index().pivot(index = 'n_epochs', columns = 'trainset')

fac_mean = fac_mean.reset_index().pivot(index = 'n_factors', columns = 'trainset')
fac_std = fac_std.reset_index().pivot(index = 'n_factors', columns = 'trainset')

plt1 = rat_mean.plot.bar(yerr=rat_std)
plt2 = ep_mean.plot.bar(yerr=ep_std)
plt3 = fac_mean.plot.bar(yerr=fac_std)

for i, p in enumerate([plt1,plt2,plt3]):
    p.set_ylabel("Average Error")
    p.legend(["(cols, 2)", "(cols, 4)", "(cols, 6)"], loc = "lower left")
    plt.savefig('..\\..\\pics for report\\cols'+str(i)+".jpg")
    



plt.show()


breakpoint()
