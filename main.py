import numpy as np
import gl
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

###############################################################
def GenerateGrid(node, xdom,ydom,numOfElx,numOfEly):

    xmin = xdom[0]; xmax = xdom[1];
    ymin = ydom[0]; ymax = ydom[1];

    dx = (xmax - xmin)/numOfElx
    dy = (ymax - ymin)/numOfEly

    x = np.zeros( [numOfElx, node.size] )
    y = np.zeros( [numOfEly, node.size] )

    for n in range(0,numOfElx):
        x[ n, : ] = n*dx + node*dx

    for n in range(0,numOfEly):
        y[ n,: ] = n*dy + node*dy
    #xx, yy = np.meshgrid(x[0,:], y[0,:])

    uinit = np.zeros([numOfElx, numOfEly, node.size, node.size])
    for ix in range(0,numOfElx):
        for iy in range(0, numOfEly):
            xx, yy = np.meshgrid(x[ix,:],y[iy,:])
            uinit[ix,iy,:,:] = np.sin(2*np.pi*xx)*np.sin(2*np.pi*yy)
            # for i in range(0, node.size):
            #     for j in range(0, node.size):
            #         uinit[ix,iy,i,j] = np.sin(2*np.pi*x[ix,i])*np.sin(2*np.pi*y[iy,j])



    return uinit,x,y,dx,dy
###############################################################

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

def ComputeCurvedFlux(detJ, defGrad, refFlux):
    # defGrad = deformation gradient of grid
    Flux = np.dot(defGrad,refFlux)/detJ
    return Flux;

def ComputeCurvedDerivatives(q,r):
    qmod = np.zeros([numOfElx, numOfEly, order+1, order+1])
    rmod = np.zeros([numOfElx, numOfEly, order+1, order+1])

    for ix in range(0,numOfElx):
        for iy in range(0,numOfEly):
            for i in range(0,order+1):
                for j in range(0, order+1):
                    refFlux = ComputeCurvedFlux( detJ, J, np.array( [q[ix,iy,i,j], r[ix,iy,i,j]] ) )
                    qmod[ix,iy,i,j] = refFlux[0];
                    rmod[ix,iy,i,j] = refFlux[1];
    return qmod, rmod

def Upwind(waveSpeed, uL,uR):
    # return 0.0
    if (waveSpeed >= 0):
        ustar = uL
    elif (waveSpeed < 0):
        ustar = uR

    return ustar
def Dupwind(waveSpeed, uL_index, uR_index):
    if (waveSpeed >= 0):
        ustar_index = uL_index;
    elif(waveSpeed < 0):
        ustar_index = uR_index;

    return ustar_index;

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
                U = np.matmul( G, u[ix,iy,i,:] )
                # Fx = detJ*invJ*Flux(beta1,U)
                Fx, Fy = ComputeReferenceFlux(detJ, invJ, beta, U)
                temp = -np.matmul( np.matmul(np.transpose(D),W), Fx)

                #print(temp)

                if (ix + 1 >= numOfElx):
                    # assuming periodic boundary conditions
                    ustar[0] = Upwind(beta[0], u[ix-1,iy,i,order], u[ix,iy,i,0])
                    ustar[1] = Upwind(beta[0], u[ix,iy,i,order], u[0,iy,i,0] )
                else:
                    ustar[0] = Upwind(beta[0], u[ix-1,iy,i,order], u[ix,iy,i,0])
                    ustar[1] = Upwind(beta[0], u[ix,iy,i,order], u[ix+1,iy,i,0])

                temp[0]     -=  ComputeReferenceFlux(detJ, invJ, beta, ustar[0])[0] #detJ*invJ[0,0]*beta1*ustar[0]
                temp[order] +=  ComputeReferenceFlux(detJ, invJ, beta, ustar[1])[0]

                q[ix,iy,i,:] = np.matmul(invMass,temp)

            # This loop computes the r = y derivative
            # Sweeps through the columns of each element
            for j in range(0,order+1):
                ustar = np.zeros(2)
                U = np.matmul(G, u[ix,iy,:,j])
                Fx, Fy = ComputeReferenceFlux(detJ, invJ, beta, U)
                temp = -np.matmul( np.matmul(np.transpose(D), W), Fy )

                if (iy + 1 >= numOfEly):
                    # periodic boundary conditions
                    ustar[0] = Upwind(beta[1], u[ix,iy-1,order,j], u[ix,iy,0,j])
                    ustar[1] = Upwind(beta[1], u[ix,iy,order,j], u[ix,0,0,j] )
                else:
                    ustar[0] = Upwind(beta[1], u[ix,iy-1,order,j], u[ix,iy,0,j] )
                    ustar[1] = Upwind(beta[1], u[ix,iy,order,j], u[ix,iy+1,0,j] )

                temp[0]     -= ComputeReferenceFlux(detJ, invJ, beta, ustar[0])[1]
                temp[order] += ComputeReferenceFlux(detJ, invJ, beta, ustar[1])[1]
                r[ix,iy,:,j] = np.matmul(invMass, temp)

    return q,r

def ComputeDt(nodes,hx, hy, waveSpeed):
    dnode_min    = np.amin(np.diff(nodes))
    h_min        = np.amin([hx,hy])
    waveSpeedMax = np.amax(waveSpeed)
    print("in computeDt")
    print(h_min, dnode_min, waveSpeedMax)

    dt = cflConst*dnode_min*h_min/waveSpeedMax;
    return dt;

def ComputeJacobian(order,totalNumOfEls, dFdu, invMassMatrix, DiffMatrix):
    Ip = np.eye(order+1)
    In = np.eye(totalNumOfEls)
    lineDx = np.matmul(invMassMatrix, -DiffMatrix)
    elementDx = np.kron(Ip,lineDx)
    elementDy = np.kron(lineDx, Ip)
    globalDx  = np.kron(In, elementDx)
    globalDy  = np.kron(In, elementDy)


    if True:
        # Implement boundary element conditions
        for ix in range(0,numOfElx):
            for iy in range(0,numOfEly):
                for i in range(0,order+1):

                    # Computes the derivative in the x-direction
                    if (ix + 1 >= numOfElx):
                        # assuming periodic boundary conditions
                        idx = Dupwind(beta[0], np.array([ix-1,iy,i,order]), np.array([ix,iy,i,0]) )
                        globalDx[Index(ix,iy,i,0),Index( idx[0],idx[1], idx[2], idx[3] )] += -1

                        idx = Dupwind(beta[0], np.array([ix,iy,i,order]), np.array([0,iy,i,0]) )
                        globalDx[ Index(ix,iy,i,order), Index(idx[0],idx[1], idx[2], idx[3] ) ] += 1
                    else:
                        idx = Dupwind(beta[0], np.array([ix-1,iy,i,order]), np.array([ix,iy,i,0]))
                        globalDx[Index(ix,iy,i,0),Index( idx[0],idx[1], idx[2], idx[3] )] += -1

                        idx = Dupwind(beta[0], np.array([ix,iy,i,order]), np.array([ix+1,iy,i,0]))
                        globalDx[Index(ix,iy,i,order), Index( idx[0],idx[1], idx[2], idx[3] )] += 1


                for j in range(0,order+1):

                    if (iy + 1 >= numOfEly):
                        #assuming periodic boundary conditions
                        idx = Dupwind(beta[1], np.array([ix,iy-1,order,j]), np.array([ix,iy,0,j]))
                        globalDy[ Index(ix,iy,0,j), Index( idx[0],idx[1], idx[2], idx[3] ) ] += 1

                        idx = Dupwind(beta[1], np.array([ix,iy,order,j]), np.array([ix,0,0,j]) )
                        globalDy[ Index(ix,iy,order,j), Index( idx[0],idx[1], idx[2], idx[3] ) ] += 1
                    else:
                        idx = Dupwind(beta[1], np.array([ix,iy-1,order,j]), np.array([ix,iy,0,j]) )
                        globalDy[ Index(ix,iy,0,j), Index( idx[0],idx[1], idx[2], idx[3] ) ] += -1

                        idx = Dupwind(beta[1], np.array([ix,iy,order,j]), np.array([ix,iy+1,0,j]) )
                        globalDy[ Index(ix,iy,order,j), Index( idx[0],idx[1], idx[2], idx[3] ) ] += 1


    globalDx  = dFdu[0]*globalDx
    globalDy  = dFdu[1]*globalDy
    globalDif = globalDx + globalDy

    if True:
        fig, ax = plt.subplots()
        ax.spy(globalDif)
        plt.show()

    return globalDif, globalDx, globalDy;

def RK4(currentTime, uold, numOfElx, numOfEly, detJ, invJ, G, D, W, beta):
    q_curved = np.zeros([numOfElx, numOfEly, order+1, order+1])
    r_curved = np.zeros([numOfElx, numOfEly, order+1, order+1])

    # Stage 1
    q,r = ComputeRHS(numOfElx, numOfEly, detJ, invJ, G, D, W, beta, uold);
    q_curved, r_curved = ComputeCurvedDerivatives(q,r)
    R   = -(q_curved + r_curved)
    k1 = dt*R

    #Stage 2
    q,r = ComputeRHS(numOfElx, numOfEly, detJ, invJ, G, D, W, beta, uold + 0.5*k1)
    q_curved, r_curved = ComputeCurvedDerivatives(q,r)
    R   = -(q_curved + r_curved)
    k2  = dt*R

    #Stage 3
    q,r = ComputeRHS(numOfElx, numOfEly, detJ, invJ, G, D, W, beta, uold + 0.5*k2);
    q_curved, r_curved = ComputeCurvedDerivatives(q,r)
    R   = -(q_curved + r_curved)
    k3  = dt*R;

    #Stage 4
    q,r = ComputeRHS(numOfElx, numOfEly, detJ, invJ, G, D, W, beta, uold + k3);
    q_curved, r_curved = ComputeCurvedDerivatives(q,r)
    R   = -(q_curved + r_curved)
    k4  = dt*R;

    unew = uold + ( k1 + 2*k2 + 2*k3 + k4)/6.0;
    return unew;

def PlotSolution(x,y,usoln):
    uplot = np.zeros([xnode[0].size*numOfElx,xnode[0].size*numOfEly])

    ax = Axes3D(fig)
    for ix in range(0,numOfElx):
        for iy in range(0,numOfEly):
            xx, yy = np.meshgrid(x[ix,:],y[iy,:])
            ax.plot_surface(xx,yy,usoln[ix,iy,:,:], rstride=1, cstride=1,edgecolor='none')
            #h = plt.contourf(xx,yy,usoln[ix,iy,:,:])

            #uplot[ix*xnode[0].size:(ix+1)*xnode[0].size, iy*xnode[0].size:(iy+1)*xnode[0].size] = usoln[ix,iy,:,:]


    plt.pause(0.05)

def Index(ielx,jely, inode, jnode):
    #need to figure out F ordering...
    index = jnode + inode*(order+1) + jely*(order+1)**2 + ielx*numOfEly*(order+1)**2;
    return index;

if __name__ == "__main__":
    # Parameters for simulation
    order = 2;
    numOfElx = 2; numOfEly = 2;
    beta1 = 1; beta2 = 1;
    beta = np.array([beta1,beta2]);
    cflConst = 0.8; tmax = 1;

    # Nodes and quadrature points
    xnode = gl.lglnodes(order)
    xint  = gl.lglnodes(3*order+1)
    W     = np.diag(xint[1])
    G,D   = gl.lagint(xnode[0], xint[0])

    # Grid "generation" and initial conditiion
    uold,x,y,dx,dy = GenerateGrid(xnode[0], np.array([0,1]), np.array([0,1]), numOfElx, numOfEly)



    J = np.diag([dx,dy])
    invJ = np.linalg.inv(J)
    detJ = np.linalg.det(J)


    # Constructing Local Operators
    Mass = np.matmul( np.matmul(np.transpose(G), W), G )
    invMass = np.linalg.inv(Mass)
    Diff  = np.matmul( np.matmul(np.transpose(D), W), G ) # This only works for linear flux

    betaTilde = detJ*np.matmul(invJ,beta)


    globalD, Dx, Dy = ComputeJacobian(order, numOfElx*numOfEly, betaTilde, invMass, Diff)

    #uold = np.arange(numOfElx*numOfEly*(order+1)*(order+1)).reshape([numOfElx,numOfEly,order+1,order+1])

    uold = np.random.rand(numOfElx,numOfEly,order+1,order+1)
    uold = np.ones([numOfElx,numOfEly,order+1,order+1])
    qmat = np.zeros( numOfElx*numOfEly*(order+1)*(order+1) )
    rmat = np.zeros( numOfEly*numOfEly*(order+1)*(order+1) )

    qmat = np.matmul(Dx, uold.ravel())
    rmat = np.matmul(Dy, uold.ravel())

    qtrue, rtrue = ComputeRHS(numOfElx, numOfEly, detJ, invJ, G, D, W, beta, uold)

    qerror = qmat - qtrue.ravel()
    rerror = rmat - rtrue.ravel()
    currentTime = 0;


    if False:
        fig = plt.figure()
        PlotSolution(x,y,uold)
        while currentTime < tmax:
            print("time = ", currentTime)
            dt = ComputeDt(xnode[0], dx, dy, beta)
            print(dt)
            if currentTime + dt > tmax:
                dt = tmax - currentTime;

            unew = RK4(currentTime, uold, numOfElx, numOfEly, detJ, invJ, G, D, W, beta)

            uold = unew;
            currentTime += dt;
            PlotSolution(x,y,uold)

        fig = plt.figure()
        PlotSolution(x,y,uold)
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.show()
    # print("q = ", q)
    # print("r = ", r)
