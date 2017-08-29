# *********************************************************
# *   tomography.py                                       *
# *                                                       *
# *   Author: Joshua Morris, josh.morris@monash.edu       *
# *                                                       *
# *   Performs tomography          *
# *********************************************************



import os
import sys
import numpy as np
import matlab.engine
import multiprocessing as mp
from itertools import product
from itertools import combinations


# gate definition dictionary - add custom gates as needed

# identity matrix
_ID = np.matrix([[1, 0], [0, 1]])
# X gate
_X = np.matrix([[0, 1], [1, 0]])
# Z gate
_Z = np.matrix([[1, 0], [0, -1]])
# Hadamard gate
_H = (1/np.sqrt(2))*np.matrix([[1, 1], [1, -1]])
# Y Gate
_Y = np.matrix([[0, -1j], [1j, 0]])
# S gate
_S = np.matrix([[1, 0], [0, 1j]])
# Sdg gate
_Sdg = np.matrix([[1, 0], [0, -1j]])
# T gate
_T = np.matrix([[1, 0], [0, (1 + 1j)/np.sqrt(2)]])
# Tdg gate
_Tdg = np.matrix([[1, 0], [0, (1 - 1j)/np.sqrt(2)]])
# CNOT gate
_CX = np.matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
# zero composition
_P1 = np.matrix([[1, 0], [0, 0]])
# one composition
_P2 = np.matrix([[0, 0], [0, 1]])

# gate dictionary
gatedict = {'h':   _H,
            'id':  _ID,
            'x':   _X,
            'y':   _Y,
            'z':   _Z,
            't':   _T,
            'tdg': _Tdg,
            's':   _S,
            'sdg': _Sdg,
            'cx':  _CX,
            'p1':  _P1,
            'p2':  _P2}


 # returns density operator series term constructed out of the Pauli
# operator basis set. Defaults to Matlab compatible output
def pauliexpand(pauli, mtlb=True):

    # initialise gate
    gate = 1.0

    for el in pauli:
        if el == '0':
            gate = np.kron(gate, gatedict['x'])
        elif el == '1':
            gate = np.kron(gate, gatedict['y'])
        elif el == '2':
            gate = np.kron(gate, gatedict['z'])
        elif el == '3':
            gate = np.kron(gate, gatedict['id'])

    if mtlb:
        return matlab.double(gate.tolist(), is_complex=True)
    else:
        return gate

# zeroes ones in binary string that contribute to normalisation condition


def stokezero(binstr, meas):
    relones = iter(index for index, item in enumerate(meas) if item == '3')
    binstr = list(binstr)
    for index in relones:
        binstr[index] = '0'
    return ''.join(binstr)



# calculates stokes parameter given measurement axis for now
def stokes(labels, probs, measureind):
    # get number of qubits
    qbits = len(measureind)
    # create dictionary using label/prob pairs
    results = {key[:-qbits-1:-1]: value for (key, value) in zip(labels, probs)}
    # create all permutations of binary string
    combs = iter(''.join(seq) for seq in product(['0', '1'], repeat=qbits))
    # iterate over all possible permutations
    stokes = 0.0
    for item in combs:
        # account for normalisation modifier
        binstr = stokezero(item, measureind)
        # probability multiplier
        if binstr.count('1') % 2 == 1:
            mult = -1
        else:
            mult = 1
        # attempt to access probability value for outcome using dictionary
        try:
            prob = mult*results[item]
        # key error for zero probability event (will only happen on simulator I
        # imagine)
        except KeyError:
            prob = 0.0
        stokes += prob

    return stokes


# useful and fast function to flatten list of lists
flatten = lambda l: [item for sublist in l for item in sublist]


# zeroes ones in binary string that contribute to normalisation condition
def stokezero(binstr, meas):
    relones = iter(index for index, item in enumerate(meas) if item == '3')
    binstr = list(binstr)
    for index in relones:
        binstr[index] = '0'
    return ''.join(binstr)



# construct identity deviation generator
def idperms(measind):
    # number of measurements to be swapped out
    replace = measind.count('2')
    replaced = 0

    # find locations of all z axis measurements
    zloc = list(index for index, item in enumerate(measind) if item == '2')
    # compute all possible permutations of swap procedure
    chk = [list(combinations(zloc, bits)) for bits in range(1, replace+1)]
    # flatten list for easy indexing
    chk = flatten(chk)
    # turn original into list for item replacement
    measind = list(measind)
    while replaced < len(chk):
        # copy original for manipulation
        yielditem = measind[:]
        for i in chk[replaced]:
            # replace iteration targets
            yielditem[i] = '3'
        # yield the new measurement sring
        yield ''.join(yielditem)
        # increment generator
        replaced += 1


# perform state tomography for a given circuit using stokes parameter
# method - no estimation, useless for anything other than perfect simulated systems
def statetomography_ST(archive, circuit):
    statecirc = archive[circuit]
    # retrieve number of qubits and hence dimensionality of system
    qubits = len(statecirc.attrs['qubits'])

    # iterate through each circuit permutation
    meas = []
    stoke = []
    for perm in statecirc:
        # skip raw qasm file
        if perm == 'raw_qasm':
            continue
        # retrieve measurement basis for permutation (order is 0:x 1:y z:2)
        meas.append(perm[-qubits:])
        label = statecirc[perm]['labels']
        probs = statecirc[perm]['values']
        stoke.append(stokes(label, probs, meas[-1]))
        # get identity stokes as well
        if '2' in meas[-1]:
            iddevs = idperms(meas[-1])
            for dev in iddevs:
                meas.append(dev)
                stoke.append(stokes(label, probs, meas[-1]))

    # construct density matrix
    density = np.zeros((2**qubits, 2**qubits), dtype='complex128')

    for index, param in enumerate(stoke):
        density += param*pauliexpand(meas[index])
    return density/(2**qubits)


# perform state tomography using maximun likelihood (requires matlab CVX)
def statetomography_ML(archive, circuit, mateng, goal='state'):
    # create container for target circuit
    statecirc = archive[circuit]
    # retrieve number of qubits and hence dimensionality of system
    qubits = len(statecirc.attrs['qubits'])

    if goal == 'state':
        # generate expectation values from dataset
        meas = []
        stoke = []
        for perm in statecirc:
            # skip raw qasm file
            if perm == 'raw_qasm':
                continue
            # retrieve measurement basis for permutation (order is 0:x 1:y z:2)
            meas.append(perm[-qubits:])
            label = statecirc[perm]['labels']
            probs = statecirc[perm]['values']
            stoke.append(stokes(label, probs, meas[-1]))
            # get identity stokes as well
            if '2' in meas[-1]:
                iddevs = idperms(meas[-1])
                for dev in iddevs:
                    meas.append(dev)
                    stoke.append(stokes(label, probs, meas[-1]))

        # map Pauli basis states into measure list and convert to Matlab
        # supported type
        meas = list(map(pauliexpand, meas[:-1]))
        # perform type conversion on stokes parameters
        stoke = list(map(float, stoke[:-1]))
        # perform MLE with Matlab CVX engine
        density = mateng.statetomography(meas, stoke)
        return density

    # perform state tomgraphy on preperation circuit sets
    if goal == 'process':
        # initialise density matrix set
        density_set = []
        # length of tomography identifier string
        strlen = 1 + 2*qubits
        # preperation state combinations
        prepcombs = product(['0', '1', '2', '3'], repeat=qubits)
        # iterate over preperation states
        for comb in prepcombs:
            # reset variables for each prep permutation
            meas = []
            stoke = []
            permstring = '_' + ''.join(str(t) for t in comb) + '_'
            for perm in statecirc:
                if perm == 'raw_qasm':
                    continue
                # perform regular state tomography on permutation
                if permstring in perm:
                    # retrieve measurement basis for permutation (order is 0:x
                    # 1:y z:2)
                    meas.append(perm[-qubits:])
                    label = statecirc[perm]['labels']
                    probs = statecirc[perm]['values']
                    stoke.append(stokes(label, probs, meas[-1]))
                    # get identity stokes as well
                    if '2' in meas[-1]:
                        iddevs = idperms(meas[-1])
                        for dev in iddevs:
                            meas.append(dev)
                            stoke.append(stokes(label, probs, meas[-1]))
            # map Pauli basis states into measure list and then convert to
            # Matlab supported type
            meas = list(map(pauliexpand, meas[:-1]))
            # perform type conversion on stokes parameters
            stoke = list(map(float, stoke[:-1]))

            # perform MLE with Matlab CVX engine
            density = mateng.statetomography(meas, stoke)
            density_set.append([permstring[:-1], density])

        return density_set


def densityplot(archive, circuit, method="ML", save=False):
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    from mpl_toolkits.mplot3d import Axes3D

    rho = archive[circuit]['tomography_' + method]

    # extract real and imaginary components of density matrix
    realrho = np.real(rho)
    imagrho = np.imag(rho)

    # instantiate new figure
    fig = plt.gcf()
    fig.canvas.set_window_title('Density Plot for circuit {}'.format(circuit))
    #rax = Axes3D(fig)
    rax = fig.add_subplot(121, projection='3d')
    iax = fig.add_subplot(122, projection='3d')

    # set titles
    rax.title.set_text('Real$(\\rho)$')
    iax.title.set_text('Imag$(\\rho)$')

    # dimension of space
    dim = np.shape(realrho)[0]
    # create indexing vectors
    x, y = np.meshgrid(range(0, dim), range(0, dim), indexing='ij')
    x = x.flatten('F')
    y = y.flatten('F')
    z = np.zeros_like(x)

    # create bar widths
    dx = 0.5*np.ones_like(z)
    dy = dx.copy()
    dzr = np.abs(realrho.flatten())
    dzi = np.abs(imagrho.flatten())

    # compute colour matrix for real matrix and set axes bounds
    norm = colors.Normalize(dzr.min(), dzr.max())
    rcolours = cm.Reds(norm(dzr))
    rax.set_zlim3d([0, dzr.max()])
    iax.set_zlim3d([0, dzr.max()])

    inorm = colors.Normalize(dzi.min(), dzi.max())
    icolours = cm.jet(inorm(dzi))

    # plot image
    rax.bar3d(x, y, z, dx, dy, dzr, color=rcolours)
    iax.bar3d(x, y, z, dx, dy, dzi, color=icolours)
    plt.ticklabel_format(style='sci', axis='z', scilimits=(0, 0))
    plt.show()


# formats a numpy array into a matlab compatible vector structure
def numpy2matlabvec(array):
    from functools import partial
    # define mappable function with complex argument
    mapmatlab = partial(matlab.double, is_complex=True)
    # perform conversion
    return list(map(mapmatlab, [item.tolist() for item in array]))


#
def numpyarr2matlab(array):
    from functools import partial
    # define mappable function with complex argument
    mapmatlab = partial(matlab.double, is_complex=True)
    # perform conversion
    return matlab.double(array.tolist(), is_complex=True)



# generates tensor products of arrayset according to combination
def kronjob(arrayset, combination):
    # initialise array
    matrix = 1
    # construct appropriate tensor product
    for el in combination:
        matrix = np.kron(matrix, arrayset[int(el)])
    return matrix



# generates preperation basis states and converts to matlab compatible
# type if requested

def prepbasis(qnum, mtlb=False):
    # define base set
    prepset = np.asarray([[[1, 0], [0, 0]],
                          [[0, 0], [0, 1]],
                          [[1/2, 1/2], [1/2, 1/2]],
                          [[1/2, -1j/2], [1j/2, 1/2]]])

    # determine all combinations
    combs = iter(''.join(seq)
                 for seq in product(['0', '1', '2', '3'], repeat=qnum))
    densitybasis = []
    # construct density basis
    for item in combs:
        densitybasis.append(kronjob(prepset, item))
    densitybasis = np.asarray(densitybasis)

    # convert to matlab compatible type if required
    if mtlb:
        return numpy2matlabvec(densitybasis)
    else:
        return densitybasis



# generates an operator spanning set for any system with qnum qubits
def opbasis(qnum, mtlb=False):
    # define operator basis set (Pauli set in this case)
    opset = np.asarray([gatedict['id'],
                        gatedict['x'],
                        gatedict['y'],
                        gatedict['z']])

    # determine all combinations
    combs = iter(''.join(seq)
                 for seq in product(['0', '1', '2', '3'], repeat=qnum))
    operbasis = []
    # construct density basis
    for item in combs:
        operbasis.append(kronjob(opset, item))
    operbasis = np.asarray(operbasis)

    # convert to matlab if required
    if mtlb:
        return numpy2matlabvec(operbasis)
    else:
        return operbasis


# define beta worker
def betawork(densitybasis, operatorbasis, partition):
    import matlab.engine
    mateng = matlab.engine.start_matlab()
    betaseg = mateng.betaopt(densitybasis, operatorbasis, partition)
    return np.asarray(betaseg)


# determines division of labor between cores
def labourdiv(dim, cores):
    # compute full workload range (+1 because of matlab indexing)
    wload = list(range(1, dim**2+1))
    # compute ~number of iterations per worker
    witer = len(wload)//cores
    # yield cores ranges until wload is empty
    for i in range(0, cores):
        # export all remaining work
        if i == cores-1:
            yield matlab.double([min(wload[i*witer:]), max(wload[i*witer:])])
        else:
            yield matlab.double([min(wload[i*witer:(i+1)*witer]), max(wload[i*witer:(i+1)*witer])])


# perform process tomography for a given circuit, assuming full measurement record
# is present
def processtomography(archive, circuit, mateng, parallel=True):
    # get number of qubits to construct basis for
    qnum = len(archive[circuit].attrs['qubits'][()])
    dim = int(2**qnum)
    # define standard basis set for CPTP map reconstruction
    # (|0>,|1>,|+>,|+i>), only single qubit at the moment
    densitybasis = prepbasis(qnum, mtlb=True)
    # define Kraus operator basis set
    operatorbasis = opbasis(qnum, mtlb=True)
    # retrieve density operators after map has acted on spanning set
    reconstruct = []
    tomogpath = archive[circuit]['tomography_ML']
    for state in tomogpath:
        reconstruct.append(matlab.double(
            tomogpath[state][()].tolist(), is_complex=True))

    # get number of available cores:
    cores = int(os.environ['NUMBER_OF_PROCESSORS'])
    # parallelise problem if requested and can be done in system
    if parallel and cores > 1 and qnum > 1:
        # It is less clear how to use Matabs inbuilt parallelisation for this problem
        # so I opted for implementing it in python. Modify as you see fit.
        print('------- DISTRIBUTING HYPERDIMENSIONAL OPTIMISATION PROBLEM ACROSS {} CORES -------'.format(cores))
        # perform labmda optimisation sequentially
        lam = mateng.lambdaopt(reconstruct, densitybasis)

        # perform beta optimisation in parallel
        # determine work division
        wrange = labourdiv(dim, cores)
        # worker list
        workers = []
        # generate worker pool
        pool = mp.Pool(processes=cores)
        
        for i, partition in enumerate(wrange):
            workers.append(pool.apply_async(betawork, args=(
                densitybasis, operatorbasis, partition)))
        pool.close()
        pool.join()
        # retrieve results
        beta = np.zeros((dim**4, dim**4), dtype='complex128')
        for employee in workers:
            beta += employee.get()

        # compute chi matrix sequentially
        chi = np.asarray(mateng.chiopt(
            numpyarr2matlab(beta), operatorbasis, lam))
        return chi, np.asarray(operatorbasis), np.asarray(densitybasis)

    else:
    	# reconstruct process matrix on single core (may be needed if RAM requirements cannot be met)
        chi = np.asarray(mateng.processtomography(
            reconstruct, densitybasis, operatorbasis))
        # output result

        return chi, np.asarray(operatorbasis), np.asarray(densitybasis)
