import numpy as np
import gl
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

def ExactSolution(xnodes, ynodes, time, beta):
    uexact = np.zeros([numOfElx, numOfEly, xnodes.shape[1], ynodes.shape[1]])
    for elx in range(numOfElx):
        for ely in range(numOfEly):
            for nodex in range(xnodes.shape[1]):
                for nodey in range(ynodes.shape[1]):
                    x = xnodes[elx, nodex]
                    y = ynodes[ely, nodey]
                    uexact[elx,ely,nodex,nodey] = np.sin(2*np.pi*(x - beta[0]*time))*np.cos(2*np.pi*(y - beta[1]*time))
    return uexact

###############################################################

def GenerateGrid(node,xdom,ydom,numOfElx,numOfEly,beta):
    xmin = xdom[0]; xmax = xdom[1]
    ymin = ydom[0]; ymax = ydom[1]

    dx = (xmax - xmin)/numOfElx
    dy = (ymax - ymin)/numOfEly

    xnodes = np.zeros( [numOfElx, node.size] )
    ynodes = np.zeros( [numOfEly, node.size] )

    for n in range(0,numOfElx):
        xnodes[ n, : ] = n*dx + node*dx

    for n in range(0,numOfEly):
        ynodes[ n,: ] = n*dy + node*dy


    uinit = ExactSolution(xnodes, ynodes, 0.0, beta)

    return uinit,xnodes,ynodes,dx,dy

####################################################################
## Computing Curved Flux and Reference Flux unstructured mesh
####################################################################
def Flux(beta, U):
    Fx = beta[0]*U
    Fy = beta[1]*U
    return Fx, Fy

def ComputeReferenceFlux(detJ, invJ, beta,U):
    Fx, Fy = Flux(beta,U)
    modFlux = np.array([np.transpose(Fx), np.transpose(Fy)])

    modFlux = detJ*np.dot(invJ,modFlux)
    if modFlux.size == 2:
        modFx = modFlux[0]
        modFy = modFlux[1]
    else:
        modFx = np.transpose(modFlux[0,:])
        modFy = np.transpose(modFlux[1,:])
    return modFx, modFy

def Upwind(waveSpeed, uL,uR):
    # return 0.0
    if (waveSpeed >= 0):
        ustar = uL
    elif (waveSpeed < 0):
        ustar = uR

    return ustar
def Dupwind(waveSpeed, uL_index, uR_index):
    if (waveSpeed >= 0):
        ustar_index = uL_index
    elif(waveSpeed < 0):
        ustar_index = uR_index
    return ustar_index

def ComputeRHS(numOfElx, numOfEly, detJ, invJ, G, D, W, beta, u):
    q = np.zeros( (numOfElx, numOfEly, order+1, order+1) )
    r = np.zeros( (numOfElx, numOfEly, order+1, order+1) )

    # Construct RHS of equation
    for ix in range(0,numOfElx):
        for iy in range(0,numOfEly):
            # This loop computes the q = x derivative
            # sweeps through the rows of each element
            for i in range(0,order+1):
                ustar = np.zeros(2)

                # Computing u and flux at the quadrature points
                U = np.matmul( G, u[ix,iy,:,i] )
                # Fx = detJ*invJ*Flux(beta1,U)
                Fx, Fy = ComputeReferenceFlux(detJ, invJ, beta, U)
                temp = -np.matmul( np.matmul(np.transpose(D),W), Fx)
                print(Fx)
                ustar[0] = Upwind(beta[0], u[ix-1,iy,-1,i], u[ix,iy,0,i])
                if (ix + 1 >= numOfElx):
                    # assuming periodic boundary conditions
                    ustar[1] = Upwind(beta[0], u[ix,iy,-1,i], u[0,iy,0,i])
                else:
                    ustar[1] = Upwind(beta[0], u[ix,iy,-1,i], u[ix+1,iy,0,i])

                temp[0]     -=  ComputeReferenceFlux(detJ, invJ, beta, ustar[0])[0]
                temp[order] +=  ComputeReferenceFlux(detJ, invJ, beta, ustar[1])[0]

                q[ix,iy,:,i] = np.matmul(invMass,temp)


            # This loop computes the r = y derivative
            # Sweeps through the columns of each element
            for j in range(0,order+1):
                ustar = np.zeros(2)
                U = np.matmul(G, u[ix,iy,j,:])
                Fx, Fy = ComputeReferenceFlux(detJ, invJ, beta, U)
                temp = -np.matmul( np.matmul(np.transpose(D), W), Fy )

                ustar[0] = Upwind(beta[1], u[ix,iy-1,j,-1], u[ix,iy,j,0])
                if (iy + 1 >= numOfEly):
                    # periodic boundary conditions
                    ustar[1] = Upwind(beta[1], u[ix,iy,j,-1], u[ix,0,j,0])
                else:
                    ustar[1] = Upwind(beta[1], u[ix,iy,j,-1], u[ix,iy+1,j,0])

                temp[0]     -= ComputeReferenceFlux(detJ, invJ, beta, ustar[0])[1]
                temp[order] += ComputeReferenceFlux(detJ, invJ, beta, ustar[1])[1]
                r[ix,iy,j,:] = np.matmul(invMass, temp)

    return q,r

def ComputeDt(p, hx, hy, waveSpeed, cfl):
    h_min        = np.amin([hx,hy])
    waveSpeedMax = np.amax(waveSpeed)
    # print("in computeDt")
    # print(h_min, dnode_min, waveSpeedMax)

    return cfl*h_min/waveSpeedMax/(p**2)

def ComputeJacobian(order,totalNumOfEls, dFdu, invMassMatrix, DiffMatrix):
    Ip = np.eye(order+1)
    In = np.eye(totalNumOfEls)
    bc = np.zeros(order+1); bc[0] = -1; bc[-1] = 1
    bc = np.dot(invMassMatrix, bc)
    lineDx = np.matmul(invMassMatrix, -DiffMatrix)
    elementDx = np.kron(lineDx, Ip)
    elementDy = np.kron(Ip, lineDx)
    globalDx  = np.kron(In, elementDx)
    globalDy  = np.kron(In, elementDy)

    if True:
        # Implement boundary element conditions
        for ix in range(0,numOfElx):
            for iy in range(0,numOfEly):
                for i in range(0,order+1):
                    # Computes the derivative in the x-direction
                    idx = Dupwind(beta[0], np.array([ix-1,iy,-1,i]), np.array([ix,iy,0,i]))
                    for ii in range(order+1):
                        globalDx[Index(ix,iy,ii,i), Index(idx[0],idx[1],idx[2],idx[3])] -= invMassMatrix[ii,0]

                    if (ix + 1 >= numOfElx):
                        # assuming periodic boundary conditions
                        idx = Dupwind(beta[0], np.array([ix,iy,-1,i]), np.array([0,iy,0,i]))
                    else:
                        idx = Dupwind(beta[0], np.array([ix,iy,-1,i]), np.array([ix+1,iy,0,i]))
                    for ii in range(order+1):
                        globalDx[Index(ix,iy,ii,i), Index(idx[0],idx[1], idx[2],idx[3])] += invMassMatrix[ii,-1]

                for j in range(0,order+1):
                    idx = Dupwind(beta[1], np.array([ix,iy-1,j,-1]), np.array([ix,iy,j,0]))
                    for ii in range(order+1):
                        globalDy[Index(ix,iy,j,ii), Index(idx[0],idx[1], idx[2], idx[3])] -= invMassMatrix[ii,0]
                    if (iy + 1 >= numOfEly):
                        #assuming periodic boundary conditions
                        idx = Dupwind(beta[1], np.array([ix,iy,j,-1]), np.array([ix,0,j,0]))
                    else:
                        idx = Dupwind(beta[1], np.array([ix,iy,j,-1]), np.array([ix,iy+1,j,0]) )
                    for ii in range(order+1):
                        globalDy[Index(ix,iy,j,ii), Index(idx[0],idx[1], idx[2], idx[3])] += invMassMatrix[ii,-1]

    globalDx  = dFdu[0]*globalDx
    globalDy  = dFdu[1]*globalDy
    globalDif = globalDx + globalDy

    if True:
        fig, ax = plt.subplots()
        ax.spy(globalDif)
        plt.show()

    return globalDif, globalDx, globalDy

def RHS(uold, numOfElx, numOfEly, detJ, invJ, G, D, W, beta):
    q,r = ComputeRHS(numOfElx, numOfEly, detJ, invJ, G, D, W, beta, uold)
    R = -(q + r)/detJ
    return R

def RK4(dt, uold, numOfElx, numOfEly, detJ, invJ, G, D, W, beta):
    q_curved = np.zeros([numOfElx, numOfEly, order+1, order+1])
    r_curved = np.zeros([numOfElx, numOfEly, order+1, order+1])

    # Stage 1
    k1 = dt*RHS(uold, numOfElx, numOfEly, detJ, invJ, G, D, W, beta)
    #Stage 2
    k2 = dt*RHS(uold + 0.5*k1, numOfElx, numOfEly, detJ, invJ, G, D, W, beta)
    #Stage 3
    k3 = dt*RHS(uold + 0.5*k2, numOfElx, numOfEly, detJ, invJ, G, D, W, beta)
    #Stage 4
    k4 = dt*RHS(uold + k3, numOfElx, numOfEly, detJ, invJ, G, D, W, beta)

    unew = uold + ( k1 + 2*k2 + 2*k3 + k4)/6.0
    return unew

# def dirk3(timeStep, currentTime, uold, order, numOfElx, numOfEly, dFdu, invMassMatrix, DiffMatrix):
#     return 0;

def PlotSolution(x,y,usoln,fig):
    uplot = np.zeros([xnode[0].size*numOfElx,xnode[0].size*numOfEly])

    ax = Axes3D(fig)
    for ix in range(0,numOfElx):
        for iy in range(0,numOfEly):
            xx, yy = np.meshgrid(x[ix,:],y[iy,:])
            ax.plot_surface(xx,yy,usoln[ix,iy,:,:].T, rstride=1, cstride=1,edgecolor='none')
            #h = plt.contourf(xx,yy,usoln[ix,iy,:,:])
            #uplot[ix*xnode[0].size:(ix+1)*xnode[0].size, iy*xnode[0].size:(iy+1)*xnode[0].size] = usoln[ix,iy,:,:]

    plt.pause(0.05)

def Index(elx, ely, nodex, nodey):
    #need to figure out F ordering...
    index = np.mod(nodey,order+1) + np.mod(nodex,order+1)*(order+1) + np.mod(ely,numOfEly)*(order+1)**2 + np.mod(elx,numOfElx)*numOfEly*(order+1)**2
    return index

if __name__ == "__main__":
    # Parameters for simulation
    order = 3
    numOfElx = 2; numOfEly = 2
    beta1 = 1; beta2 = -2
    beta = np.array([beta1,beta2])
    cflConst = 0.8; tmax = 0.25

    print("Order = ", order)
    print("Nx = ", numOfElx, " Ny = ", numOfEly)
    print("CFL = ", cflConst)
    print("Beta = ", beta1, beta2)
    print()

    # Nodes and quadrature points
    xnode = gl.lglnodes(order)
    xint  = gl.lglnodes(3*order+1)
    W     = np.diag(xint[1])
    G,D   = gl.lagint(xnode[0], xint[0])

    # Grid "generation" and initial conditiion
    uold,x,y,dx,dy = GenerateGrid(xnode[0], np.array([0,1]), np.array([0,1]), numOfElx, numOfEly, beta)

    J = np.diag([dx,dy])
    invJ = np.linalg.inv(J)
    detJ = np.linalg.det(J)

    # # Constructing Local Operators
    Mass = np.matmul( np.matmul(np.transpose(G), W), G )
    invMass = np.linalg.inv(Mass)
    Diff  = np.matmul( np.matmul(np.transpose(D), W), G ) # This only works for linear flux

    betaTilde = detJ*np.matmul(invJ,beta)

    globalD, Dx, Dy = ComputeJacobian(order, numOfElx*numOfEly, betaTilde, invMass, Diff)

    #uold = np.random.rand(numOfElx,numOfEly,order+1,order+1)
    # uold = np.ones([numOfElx,numOfEly,order+1,order+1])

    qmat = np.zeros( numOfElx*numOfEly*(order+1)*(order+1) )
    rmat = np.zeros( numOfEly*numOfEly*(order+1)*(order+1) )

    qmat = np.matmul(Dx, uold.ravel())
    rmat = np.matmul(Dy, uold.ravel())

    qtrue, rtrue = ComputeRHS(numOfElx, numOfEly, detJ, invJ, G, D, W, beta, uold)

    qerror = qmat - qtrue.ravel()
    rerror = rmat - rtrue.ravel()

    print("max(| q_error |) = ", np.amax(np.abs(qerror)))
    print("max(| r_error |) = ", np.amax(np.abs(rerror)))

    fig = plt.figure()
    shp = [numOfElx,numOfEly,order+1,order+1]
    #PlotSolution(x,y,(qmat+rmat).reshape(shp),fig)
    #PlotSolution(x,y,(qtrue+rtrue).reshape(shp),fig)
    #PlotSolution(x,y,qtrue-qmat.reshape(shp),fig)

    # r = RHS(uold, numOfElx, numOfEly, detJ, invJ, G, D, W, beta)
    currentTime = 0
    # while currentTime < tmax:
    #     print("time = ", currentTime)
    #     dt = ComputeDt(order, dx, dy, beta, cflConst)
    #     # print("time step = ", dt)
    #
    #     if currentTime + dt > tmax:
    #         dt = tmax - currentTime
    #
    #     unew = RK4(dt, uold, numOfElx, numOfEly, detJ, invJ, G, D, W, beta)
    #     uold = unew
    #     currentTime += dt
    #     print("max( |unew| ) = ", np.amax(np.abs(unew)))
    #     PlotSolution(x,y,unew, fig)

    # uex  = ExactSolution(x,y,currentTime,beta)
    # uerror = np.amax( np.abs( uold -  uex) )
    # print("max(| uerror|) = ", uerror)
