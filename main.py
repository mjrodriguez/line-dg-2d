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
    if (waveSpeed > 0):
        ustar = uL
    elif (waveSpeed < 0):
        ustar = uR

    return ustar

def ComputeRHS(u):

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
                    # assuming periodic boundary conditions
                    ustar[0] = Upwind(beta[1], u[ix,iy-1,order,j], u[ix,iy,0,j])
                    ustar[1] = Upwind(beta[1], u[ix,iy,order,j], u[ix,0,0,j] )
                else:
                    ustar[0] = Upwind(beta[1], u[ix,iy-1,order,j], u[ix,iy,0,j] )
                    ustar[1] = Upwind(beta[1], u[ix,iy,order,j], u[ix,iy+1,0,j] )

                temp[0]     -= ComputeReferenceFlux(detJ, invJ, beta, ustar[0])[1]
                temp[order] += ComputeReferenceFlux(detJ, invJ, beta, ustar[1])[1]

                r[ix,iy,:,j] = np.matmul(invMass, temp)

    return q,r

def ComputeDt():
    dnode_min    = np.amin(np.diff(xnode[0]))
    h_min        = np.amin([dx,dy])
    waveSpeedMax = np.amax([beta1,beta2])

    dt = cflConst*dnode_min*h_min/waveSpeedMax;
    return dt;

def ComputeJacobian(order,totalNumOfEls, beta, invMassMatrix, DiffMatrix):
    Ip = np.eye(order+1)
    In = np.eye(totalNumOfEls)
    lineDx = np.matmul(invMassMatrix,-DiffMatrix)
    elementDx = np.kron(Ip,lineDx)
    elementDy = np.kron(lineDx, Ip)
    globalDx  = detJ*beta[0]*np.kron(In, elementDx)/dx
    # elementDy = np.diag(np.diag(elementDx))
    #
    # for i in range(1,order+1):
    #     du = np.diag(elementDx,k=i)
    #     db = np.diag(elementDx, k=-i)
    #     elementDy += np.diag(du[du!=0],k=i*(order+1))
    #     elementDy += np.diag(db[db!=0],k=-i*(order+1))

    globalDy = detJ*beta[1]*np.kron(In, elementDy)/dy
    #print("global dy = ", globalDy)
    globalDif = globalDx + globalDy
    fig, ax = plt.subplots()
    ax.spy(globalDif)
    plt.show()

    return globalDx, globalDy;

def RK4(currentTime, uold):
    q_curved = np.zeros([numOfElx, numOfEly, order+1, order+1])
    r_curved = np.zeros([numOfElx, numOfEly, order+1, order+1])

    # Stage 1
    q,r = ComputeRHS(uold);
    q_curved, r_curved = ComputeCurvedDerivatives(q,r)
    R   = -(q_curved + r_curved)
    k1 = dt*R

    #Stage 2
    q,r = ComputeRHS(uold + 0.5*k1)
    q_curved, r_curved = ComputeCurvedDerivatives(q,r)
    R   = -(q_curved + r_curved)
    k2  = dt*R

    #Stage 3
    q,r = ComputeRHS(uold + 0.5*k2);
    q_curved, r_curved = ComputeCurvedDerivatives(q,r)
    R   = -(q_curved + r_curved)
    k3  = dt*R;

    #Stage 4
    q,r = ComputeRHS(uold + k3);
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


if __name__ == "__main__":
    # Parameters for simulation
    order = 2;
    numOfElx = 2; numOfEly = 2;
    beta1 = 1; beta2 = 1;
    beta = np.array([beta1,beta2]);
    cflConst = 1.0; tmax = 0.5;

    # Nodes and quadrature points
    xnode = gl.lglnodes(order)
    xint  = gl.lglnodes(3*order+1)
    W     = np.diag(xint[1])
    G,D   = gl.lagint(xnode[0], xint[0])

    # Grid "generation" and initial conditiion
    uold,x,y,dx,dy = GenerateGrid(xnode[0], np.array([0,1]), np.array([0,1]), numOfElx, numOfEly)

    # fig = plt.figure()
    # PlotSolution(x,y,uold)

    J = np.diag([dx,dy])
    invJ = np.linalg.inv(J)
    detJ = np.linalg.det(J)


    # Constructing Local Operators
    Mass = np.matmul( np.matmul(np.transpose(G), W), G )
    invMass = np.linalg.inv(Mass)
    Diff  = np.matmul( np.matmul(np.transpose(D), W), G ) # This only works for linear flux

    Dx, Dy = ComputeJacobian(order, numOfElx*numOfEly, beta, invMass, Diff)

    #uold = np.arange(numOfElx*numOfEly*(order+1)*(order+1)).reshape([numOfElx,numOfEly,order+1,order+1])
    uold = np.ones([numOfElx,numOfEly,order+1,order+1])

    qmat = np.zeros( numOfElx*numOfEly*(order+1)*(order+1) )
    rmat = np.zeros( numOfEly*numOfEly*(order+1)*(order+1) )

    qmat = np.matmul(Dx, uold.reshape([numOfElx*numOfEly*(order+1)*(order+1)]))
    rmat = np.matmul(Dy, uold.reshape([numOfElx*numOfEly*(order+1)*(order+1)]))

    q = np.zeros( (numOfElx, numOfEly, order+1, order+1) )
    r = np.zeros( (numOfEly, numOfEly, order+1, order+1) )

    q,r = ComputeRHS(uold);

    currentTime = 0;


    # q,r = ComputeRHS(uold);
    #
    #
    #
    # while currentTime < tmax:
    #     # print("time = ", currentTime)
    #     dt = ComputeDt()
    #     if currentTime + dt > tmax:
    #         dt = tmax - currentTime;
    #
    #     unew = RK4(currentTime, uold)
    #
    #     uold = unew;
    #     currentTime += dt;
    #
    #     #PlotSolution(x,y,uold)
    #
    # fig = plt.figure()
    # PlotSolution(x,y,uold)
    # plt.xlabel('$x$')
    # plt.ylabel('$y$')
    # plt.show()
    # print("q = ", q)
    # print("r = ", r)
