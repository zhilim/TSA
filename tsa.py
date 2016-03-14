import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.graphics.tsaplots as tp 
import psotest as pso 
import statsmodels.tsa.arima_model as ap
import math
import statsmodels as sm



#load s&p500 price time series from csv
def loadIndex(stationary):
	raw = pd.DataFrame.from_csv('^GSPC.csv')
	if stationary:
		chiseled = np.diff(np.log(raw['Adj Close'][:'1990'].iloc[::-1]))
	elif not stationary:
		chiseled = raw['Adj Close'][:'1990'].iloc[::-1]
	return list(chiseled)

#load global active power time series from data file
#does daily, monthly, hourly, and by minute, but we only use daily
def loadPower():
	f = open('household_power_consumption.txt', 'r')
	plainseries = []
	dailyseries = []
	monthlyseries = []
	hourly = []
	dailyavg = []
	monthlyavg = []
	hourlyavg = []
	dateNow = -1
	monthNow = -1
	timeNow = -1
	print "LOADING POWER CONSUMPTION DATA"
	for line in f.readlines():
		ls = line.split(';')
		if ls[0] == 'Date':
			continue
		try:
			series.append(float(ls[2]))
		except:
			pass
			#print "Missing value, skipping over\n"
		#print ls[0]
		date = ls[0].split('/')
		time = ls[1].split(':')
		#print date[0], date[1], date[2]

		if ls[2] == None or ls[2] == '?':
			continue

		if float(time[0]) == timeNow or timeNow == -1:
			hourlyavg.append(float(ls[2]))
			timeNow = float(time[0])

		if float(time[0]) != timeNow:
			hourly.append(np.mean(hourlyavg))
			hourlyavg = []
			timeNow = float(time[0])

		if float(date[0]) == dateNow or dateNow == -1:
			dailyavg.append(float(ls[2]))
			dateNow = float(date[0])

		if float(date[0]) != dateNow:
			dailyseries.append(np.mean(dailyavg))
			dailyavg = []
			dateNow = float(date[0])

		if float(date[1]) == monthNow or monthNow == -1:
			monthlyavg.append(float(ls[2]))
			monthNow = float(date[1])

		if float(date[1]) != monthNow:
			monthlyseries.append(np.mean(monthlyavg))
			monthlyavg = []
			monthNow = float(date[1])

	print "number of days: " + str(len(dailyseries))
	print "number of months: " + str(len(monthlyseries))
	print "number of hours: " + str(len(hourly))

	return plainseries, monthlyseries, dailyseries, hourly



#main PSO predictor class. Contains cost function methods and prediction methods
class Prophet:
	series = []
	dimensions = 0
	std = 0
	mean = 0
	weights = []

	def __init__(self, x, d):
		self.series = x
		self.std = np.std(self.series)
		self.mean = np.mean(self.series)
		self.dimensions = d

	#cost function: Mean squared error
	def divinecost(self, w):
		predictedSeries = [0] * self.dimensions
		squaredErrors = []
		for t in range(self.dimensions, len(self.series)):
			xtpredicted = 0
			for r in range(self.dimensions):
				xtpredicted += w[r] * self.series[t-(r+1)]
			#xtpredicted += xtpredicted - predictedSeries[t-1]
			predictedSeries.append(xtpredicted)
			sqerr = (self.series[t] - xtpredicted)**2
			squaredErrors.append(sqerr)
		#pocid = POCID(self.series[self.dimensions:], predictedSeries[self.dimensions:])

		mse = np.mean(squaredErrors)

		#cost = 0.8*mse + 0.2*(100-pocid)
		return mse

	#predicts values for 'length' number of steps ahead of last day of time series
	#returns inSampleP = in sample prediction, fromOrginalSeries = insample concatenate outsample prediction
	# outsamplep = outsample prediction
	def prophesize(self, length):
		inSampleP = [self.mean] * self.dimensions
		for t in range(self.dimensions, len(self.series)):
			xtpredicted = 0
			for r in range(self.dimensions):
				xtpredicted += self.weights[r] * self.series[t-(r+1)]
			#xtpredicted += xtpredicted - inSampleP[t-1]
			inSampleP.append(xtpredicted)

		fromOriginalSeries = list(self.series)
		outSampleP = []
		for ft in range(len(self.series), len(self.series)+length):
			xtpredicted = 0
			for r in range(self.dimensions):
				xtpredicted += self.weights[r] * fromOriginalSeries[ft-(r+1)]
			#xtpredicted += np.random.normal(loc=0.0, scale=self.std)
			#xtpredicted += (xtpredicted - inSampleP[t-1])*0.5
			fromOriginalSeries.append(xtpredicted)
			outSampleP.append(xtpredicted)
			inSampleP.append(xtpredicted)

		return inSampleP, fromOriginalSeries, outSampleP

	#fits a model to the series, uses cost function as fitness function for particle swarm
	#returns s = vector of coefficients, i = number of iterations, g = lowest cost reached
	def fit(self):
		pso.dmn = self.dimensions
		pso.searchRange = float(1)/float(pso.dmn)
		pso.maxIterations = 100
		s,i,g = pso.particleSwarmOptimize(self.divinecost, True, True)
		self.weights = s
		return s,i,g


#experimental PSO implementation of ARIMA process
#deprecated, do not use
class Arima:
	ar = []
	ma = []
	error = 0
	series = []
	std = 0
	mean = 0
	p = 0
	d = 0
	q = 0
	dimensions = 0
	varC = 0
	wnC = 0

	def __init__(self, p, d, q, x):
		self.ar = [0] * p
		self.ma = [0] * q
		self.series = x
		for i in range(d):
			self.series = np.diff(self.series)
		self.p = p
		self.q = q
		self.d = d
		self.std = np.std(self.series)
		self.dimensions = p+q
		self.mean = np.mean(self.series)

	def costfunction(self, coeffs):

		arc = coeffs[:self.p]
		mac = coeffs[self.p:self.p+self.q]
		#varc = coeffs[self.p+self.q]
		#wnc = coeffs[self.p+self.q+1]
		#errorc = coeffs[self.p+self.q]

		predictedSeries = []
		errorTerms = []
		errorTerms.append(0)
		predictedSeries.append(0)
		for t in range(1,len(self.series)):
			xtpredicted = 0
			if t < self.p or t < self.q:
				if t < self.p:
					for r in range(t):
						xtpredicted += arc[r] * self.series[t-(r+1)] 
				elif t >= self.p:
					for r in range(self.p):
						xtpredicted += arc[r] * self.series[t-(r+1)] 
				if t < self.q:
					rnge = range(t)
				elif t >= self.q:
					rnge = range(self.q)
				for r in rnge:
					xtpredicted += mac[r] * errorTerms[(len(errorTerms)-1)-r]
				#xtpredicted = xtpredicted * varc * (self.std**2)
				xtpredicted += np.random.normal(loc=0.0, scale=self.std)
				errorTerms.append(self.series[t] - xtpredicted)
				predictedSeries.append(xtpredicted)
			else:
				for r in range(self.p):
					xtpredicted += arc[r] * self.series[t-(r+1)]
				for r in range(self.q):
					xtpredicted += mac[r] * errorTerms[(len(errorTerms)-1)-r]
				#xtpredicted = xtpredicted * varc * (self.std**2)
				xtpredicted += np.random.normal(loc=0.0, scale=self.std)
				errorTerms.append(self.series[t] - xtpredicted)
				predictedSeries.append(xtpredicted)

		tradecost = 0
		wrongcalls = 0
		distancecost = 0

		for i in range(1,len(self.series)):
			if self.series[i] > 0 and predictedSeries[i] < 0:
				wrongcalls += 1
			if self.series[i] < 0 and predictedSeries[i] > 0:
				wrongcalls += 1
			tradecost = float(wrongcalls)/float((len(self.series)-1))

		for e in errorTerms:
			distancecost += e**2
		distancecost = float(distancecost)/float(len(self.series))
		return distancecost

	def predict(self, future):
		arc = self.ar
		mac = self.ma
		#varc = self.varC
		#wnc = self.wnC
		predictedSeries = []
		errorTerms = []
		errorTerms.append(0)
		predictedSeries.append(0)
		for t in range(1,len(self.series)):
			xtpredicted = 0
			if t < self.p or t < self.q:
				if t < self.p:
					for r in range(t):
						xtpredicted += arc[r] * self.series[t-(r+1)] 
				elif t >= self.p:
					for r in range(self.p):
						xtpredicted += arc[r] * self.series[t-(r+1)] 
				if t < self.q:
					rnge = range(t)
				elif t >= self.q:
					rnge = range(self.q)
				for r in rnge:
					xtpredicted += mac[r] * errorTerms[(len(errorTerms)-1)-r]
				#xtpredicted = xtpredicted * varc * (self.std**2)
				xtpredicted += np.random.normal(loc=0.0, scale=self.std)
				errorTerms.append(self.series[t] - xtpredicted)
				predictedSeries.append(xtpredicted)
			else:
				for r in range(self.p):
					xtpredicted += arc[r] * self.series[t-(r+1)]
				for r in range(self.q):
					xtpredicted += mac[r] * errorTerms[(len(errorTerms)-1)-r]
				#xtpredicted = xtpredicted * varc * (self.std**2)
				xtpredicted += np.random.normal(loc=0.0, scale=self.std)
				errorTerms.append(self.series[t] - xtpredicted)
				predictedSeries.append(xtpredicted)

		cost = 0
		for e in errorTerms:
			cost += e**2

		fromOriginalSeries = list(self.series)

		for ft in range(future):
			xtpredicted = 0
			for r in range(self.p):
				xtpredicted += arc[r] * fromOriginalSeries[t-(r+1)]
			for r in range(self.q):
				xtpredicted += mac[r] * errorTerms[(len(errorTerms)-1)-r]
			#xtpredicted = xtpredicted * varc * (self.std**2)
			xtpredicted += np.random.normal(loc=0.0, scale=self.std)
			errorTerms.append(fromOriginalSeries[t] - xtpredicted)
			predictedSeries.append(xtpredicted)
			fromOriginalSeries.append(xtpredicted)

		return cost, errorTerms, predictedSeries, fromOriginalSeries

	def fit(self):
		pso.dmn = self.dimensions
		s, i, g = pso.particleSwarmOptimize(self.costfunction, True, True)
		self.ar = s[:self.p]
		self.ma = s[self.p:self.p+self.q]
		#self.varC = s[self.p+self.q]
		#self.wnC = s[self.p+self.q+1]
		#self.error = s[self.p+self.q]
		print s, i, g
		return s, i, g

#deprecated, do not use
def calculateFPrate(actual, predicted):
	wc = 0
	mean = np.mean(predicted)
	for i in range(len(actual)):
		if actual[i] > 0 and predicted[i] < 0:
			wc += 1
		if actual[i] < 0 and predicted[i] > 0:
			wc += 1
	return float(wc)/float(len(actual))

#deprecated, do not use
def armafit(order, d):
	lowestcost = 999999
	bestarma = None
	for p in range(1, order):
		for q in range(1, order):
			arma = Arima(p,0,q,d)
			s,i,g = arma.fit()
			if g < lowestcost:
				bestarma = arma
				lowestcost = g

	return bestarma

#deprecated, do not use
def windowTest(window, end):
	d = loadIndex()
	wrongcalls = 0
	prediction = []
	for i in range(end-window-1):
		trainer = d[i:window+i]
		a = armafit(4, trainer)
		c,e,p,o = a.predict(1)
		if d[window+i] > 0 and p[window] < 0:
			wrongcalls += 1
		if d[window+i] < 0 and p[window] > 0:
			wrongcalls += 1
		prediction.append(p[window])
	plt.plot(d[window:end])
	plt.plot(prediction)
	plt.savefig("accuracy.png")
	fprate = float(wrongcalls)/float(end-window-1)
	print fprate
	return fprate, prediction

#do not use, developer's test kit
def test():
	d = loadIndex()
	std = np.std(d)
	print len(d), std

	a = Arima(1,0,1,d[:500])
	print a.std
	c, e, p = a.costfunction([0.1, 0.9])
	print c, len(e), len(p)
	c,e,p,o = a.predict(25)
	plt.plot(d[200:225])
	plt.plot(o)
	plt.show()	

#calculate and print metrics to log
def metrics(actual, prediction, inActual, inPredict, filename, iden):
	f = open(filename, 'a')
	inMape = MAPE(inActual, inPredict)
	outMape = MAPE(actual, prediction)
	inPoc = POCID(inActual, inPredict)
	outPoc = POCID(actual, prediction)
	inTheil = THEIL(inActual, inPredict)
	outTheil = THEIL(actual, prediction)
	rep =  iden + "," + str(inMape) + "," + str(outMape) + "," + str(inPoc) + "," + str(outPoc) + "," + str(inTheil) + "," + str(outTheil) + "\n"
	f.write(rep)
	f.close()

#predicts, plots graphs, calculates metrics, using ARMA process for @param prsize steps ahead, 
#@param periods number of periods, @param trsize = size of training data, @param fullDataSet = time series, 
#@param folder = folder to send results to
def arimaControl(periods, trsize, prsize, fullDataSet, folder):
	d = fullDataSet
	prediction = []
	inSamp = []

	for period in range(periods):
		dcut = d[period*prsize:trsize+(period*prsize)]
		lowestAIC = 999999
		roleModel = None
		for p in range(4):
			for q in range(4):
				try:
					arma = ap.ARMA(dcut, (p,q)).fit()
				except:
					pass
				if arma.aic < lowestAIC:
					lowestAIC = arma.aic
					roleModel = arma
		prediction += list(roleModel.predict(trsize, trsize+prsize-1))
		if period == 0:
			inSamp = list(roleModel.predict(0,trsize-1))
	actual = d[trsize:trsize+(periods*prsize)]
	inActual = d[:trsize]

	plt.clf()
	plt.plot(actual, label='actual')
	plt.plot(prediction, label='ARMA')
	#plt.title("ARMA Process Control Set")
	#plt.ylabel("Global Active Power")
	#plt.xlabel("Days")
	plt.legend(loc=2)
	iden = "ARMAstep" + str(prsize) + "trsize" + str(trsize) + "p" + str(periods)
	plt.savefig(folder+'/arma_' + iden + '.png')

	#plt.clf()
	#plt.plot(inActual+actual, label='actual')
	#plt.plot(inSamp+prediction, label='ARMA')
	#plt.title("In-sample ARMA Control Set: Power Daily")
	#plt.ylabel("Global Active Power")
	#plt.xlabel("Days")
	#plt.legend(loc=2)
	#plt.savefig(folder+'/inArma_' + iden + '.png')
	cum = backtest(actual, prediction)
	bm = benchmark(actual)
	plt.plot(cum, label='ARMA')
	plt.plot(bm, label='sp500')
	plt.legend(loc=2)
	plt.savefig(folder+'/benchmark.png')


	
	metrics(actual, prediction, inActual, inSamp, folder+'/log.txt', iden)


#predicts, plots graphs, calculates metrics for PSO predictions
def psoTest(periods, trsize, prsize, dayAhead, fullDataSet, folder):
	

	d = fullDataSet
	prediction = []
	orig = []

	if dayAhead:
		for period in range(periods):
			dcut = d[period*prsize:trsize+(period*prsize)]

			if prsize <= 5:
				dimns = 5
			else:
				dimns = prsize
			a = Prophet(dcut, 100)
			a.fit()
			for t in range(prsize):
				newcut = d[period*prsize+t:trsize+(period*prsize)+t]
				a.series = newcut
				inSample, fromo, outSample = a.prophesize(1)
				prediction += outSample
				if period == 0 and t == 0:
					orig = inSample[:trsize]
	elif not dayAhead:
		for period in range(periods):
			dcut = d[period*prsize:trsize+(period*prsize)]

			if prsize <= 5:
				dimns = 5
			else:
				dimns = prsize
			a = Prophet(dcut, 1500)
			a.fit()
			inSample, fromo, outSample = a.prophesize(prsize)
			prediction += outSample
			if period == 0:
				orig = inSample[:trsize]
		
	#print len(prediction)
	#print len(d[trsize:trsize+(periods*prsize)])
	actual = d[trsize:trsize+(periods*prsize)]
	inActual = d[:trsize]
	inSamp = orig[:trsize]
	naive = randomWalk(d[trsize-1:trsize+(periods*prsize)])

	iden = "PSOstep" 
	if dayAhead:
		iden += str(1) + "trsize" + str(trsize) + "p" + str(prsize*periods)
	elif not dayAhead:
		iden += str(prsize) + "trsize" + str(trsize) + "p" + str(periods)
	metrics(actual, prediction, inActual, inSamp, folder+'/log.txt', iden)

	plt.plot(d[trsize:trsize+(periods*prsize)], label='actual')
	plt.plot(prediction, label='pso')
	#plt.plot(naive, label='random walk')
	plt.legend(loc=2)
	plt.savefig(folder + '/pso_' + iden + '.png')
	#cum = backtest(d[trsize:trsize+(periods*prsize)], prediction)
	#bm = benchmark(d[trsize:trsize+(periods*prsize)])
	plt.clf()
	plt.plot(d[:trsize+(periods*prsize)], label='actual')
	plt.plot(orig+prediction, label='pso')
	plt.legend(loc=2)
	plt.savefig(folder+'/inPso_' + iden +'.png')
	plt.clf()
	#plt.plot(cum, label='pso')
	#plt.plot(bm, label='sp500')
	#plt.legend(loc=2)
	#plt.savefig('benchmark.png')

#randomwalk simulation
def randomWalk(dcut):
	prediction = []
	for i in range(1,len(dcut)):
		prediction.append(dcut[i-1] + np.random.normal(loc=0, scale=np.std(dcut)))

	return prediction

#backtest for S&P500
#barebones, no transaction costs simulation
#no drawdown, alpha, beta, or sortino calculations
def backtest(actual, predicted):
	mean = np.mean(predicted)
	cum = []
	total = 1
	for i in range(len(actual)):
		if actual[i] > 0 and predicted[i] < mean:
			total = total * (1-math.fabs(actual[i]))
		if actual[i] < 0 and predicted[i] > mean:
			total = total * (1-math.fabs(actual[i]))
		if actual[i] > 0 and predicted[i] > mean:
			total = total * (1+math.fabs(actual[i]))
		if actual[i] < 0 and predicted[i] < mean:
			total = total * (1+math.fabs(actual[i]))
		cum.append((total-1)*100)
	return cum

#calculate cumulative return of S&P500
def benchmark(actual):
	cum = []
	total = 1
	for ret in actual:
		total = total * (1+ret)
		cum.append((total-1)*100)
	return cum

#calculate MAPE
def MAPE(actual, predicted):
	summ = 0
	for i in range(len(actual)):
		if actual[i] != 0:
			summ += math.fabs((float(actual[i]) -float(predicted[i]))/ float(actual[i]))
	return float(summ)/float(len(actual))

#Calculate POCID
def POCID(actual, predicted):
	dSum = 0
	for j in range(1, len(actual)):
		if ((actual[j] - actual[j-1])*(predicted[j] - predicted[j-1])) > 0:
			dSum += 1
	pocid = 100*(float(dSum)/float((len(actual)-1)))
	return pocid

#Calculate MAE
def MAE(actual, predicted):
	se = 0
	for i in range(len(actual)):
		se += math.fabs(actual[i] - predicted[i])
	return np.mean(se)

#Calculate MASE
def THEIL(actual, predicted):
	rw = randomWalk(actual)
	rwmae = MAE(actual[1:], rw)
	pmae = MAE(actual, predicted)
	return pmae/rwmae


#plot acf and raw time series 
def preliminaries(data):
	plt.plot(data)
	title = "Daily Time Series for S&P 500"
	plt.title(title)
	plt.ylabel("Price")
	plt.xlabel("Day")
	fn = "results/snp.png"
	plt.savefig(fn)

	acf = tp.plot_acf(data)
	acf.savefig("results/snpacf.png")


'''
TESTING BEGINS HERE
'''

#d = sm.datasets.sunspots.load_pandas().data
#d = list(d['SUNACTIVITY'])
#p, m, d, h = loadPower()
#preliminaries(np.diff(np.log(d)))
#d = np.diff(np.log(d))
d = loadIndex(True)
#preliminaries(d)
#d = loadPower()
#plt.plot(d)
#plt.title('Raw Plot of Time Series')
#plt.savefig('raw.png')
#plt.clf()
#psoTest(1, 200, 20, False, d, 'sunspots_results')
arimaControl(2500,500,1,d,'results')
#psoTest(1,200,20,False,d,'powerstat')
#psoTest(20,200,1,False,d,'powerstat')
#psoTest(1,200,20,True,d,'results')




#test()
#windowTest(200, 205)

'''

arparams = np.array([0.1, 0.1])
maparams = np.array([0.1, 0.2])
ar = np.r_[1, -arparams]
ma = np.r_[1, maparams]

y = loadIndex()


a = Prophet(y[:500], 20)
a.fit()
inSample, fromo, outSample = a.prophesize(5)
plt.plot(y[500:505])
plt.plot(outSample)
print calculateFPrate(y[500:505], outSample)
print np.mean(y[500:505])
print np.mean(outSample)
print np.std(y[500:505])
print MAPE(y[500:505], outSample)
plt.show()

'''













