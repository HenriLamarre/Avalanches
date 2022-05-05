!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! SOC_Case = 100 in python !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!
subroutine do_avalanche_generic(Niter, &                 ! INPUT
     Nx,Ny,Binit,Zc,D_nc,eps_drive,sigma1,sigma2,lh_soc,idum_init,verbose, &     ! OPTIONAL
     lattice_energy,energy_released,energy_released2,B,Z)         ! OUTPUTS
  use permutations
  use Prec
  implicit none
  Integer    :: Nx,Ny
!f2py integer optional,intent(in) :: Nx=16
!f2py integer optional,intent(in) :: Ny=16
  Integer    :: Niter
!f2py intent(in) Niter
  Real(mp) :: Zc,D_nc,eps_drive,sigma1,sigma2
!f2py real(mp) optional, intent(in) :: Zc = 1.0
!f2py real(mp) optional, intent(in) :: D_nc=0.98
!f2py real(mp) optional, intent(in) :: eps_drive=1.e-7
!f2py real(mp) optional, intent(in) :: sigma1=0.0
!f2py real(mp) optional, intent(in) :: sigma2=0.0
  Real(mp) :: lh_soc
!f2py real(mp) optional,intent(in) :: lh_soc=0.0
  Integer :: idum_init
!f2py integer optional, intent(in) :: idum_init = -67209495
  Real(mp) :: lattice_energy(Niter),energy_released(Niter),energy_released2(Niter)
!f2py intent(out) lattice_energy
!f2py intent(out) energy_released
!f2py intent(out) energy_released2
!f2py depend(Niter) lattice_energy
!f2py depend(Niter) energy_released
!f2py depend(Niter) energy_released2
  Real(mp) :: B(Nx,Ny)
!f2py intent(out) B
!f2py depend(Nx,Ny) B
  Real(mp) :: Z(Nx,Ny)
!f2py intent(out) Z
!f2py depend(Nx,Ny) Z
  Real(mp) :: Binit(Nx,Ny)
!f2py real(mp) optional, intent(in), depend(Nx,Ny) :: Binit = 0.0
  Logical :: verbose
!f2py logical optional, intent(in) :: verbose=0

  ! Local variables
  Integer   :: i=1,j=1,iter=1
  Real(mp)  :: increment=0.1,energy_rel=0.1,old_energy=0.1
  Real(mp)  :: energy2=0.1,energy_rel2=0.1
  Integer   :: niter_to_do=1,iter_tmp=1
  Logical   :: first_node = .true., first_node_rdist=.true.
  Real(mp)  :: D=2.0,s=5.0,e0=1.0,my_rand=1.0,gauss_rand,my_rand_i=1.0,my_rand_j=1.0,deltaB=0.1,meanB=1.0
  Real(mp)  :: ZZ=1.0,r_dnc=1.0,r0=1.0,r1=1.0,r2=1.0,r3=1.0,rr=1.0,energy=1.0,Zc_loc=1.0
  Real(mp)  :: r0_tmp=1.0,r1_tmp=1.0,r2_tmp=1.0,r3_tmp=1.0
  Real*4    :: ran2
  Integer   :: idum_run=-1,last_iter_oneiter=1,itran=1,itrmax=100
  Real(mp), dimension(:,:), allocatable :: C

  Real(mp)  :: zero,one,two,four

  last_iter_oneiter = -10

  ! Initialize important variables
  zero = real(0,mp)
  one  = real(1,mp)
  two  = real(2,mp)
  four = real(4,mp)
  itrmax=100
  first_node = .true.
  first_node_rdist = .true.

  ! For random numbers
  idum_run = idum_init

  ! Allocate tables
  Allocate(C(Nx,Ny))
  C(:,:) = zero

  D  = two
  s  = two*D+one
  e0 = two*D*(Zc**2)/s

  B(:,:) = Binit
  ! Enforce B = 0 on boundaries
  B(1,:)  = 0._mp
  B(Nx,:) = 0._mp
  B(:,1)  = 0._mp
  B(:,Ny) = 0._mp
  Z(:,:)  = zero
  do i = 2,Nx-1
     do j = 2,Ny-1
        Z(i,j) = B(i,j) - (B(i+1,j)+B(i-1,j)+B(i,j+1)+B(i,j-1))/(two*D)
     end do
  end do


  iter = 1
  ! Main Loop
  do while (iter .le. Niter)

     energy =zero
     energy2=zero
     first_node=.true.
     first_node_rdist = .true.
     ! Compute curvature
     do i = 2,Nx-1
        do j = 2,Ny-1
           Z(i,j) = B(i,j) - (B(i+1,j)+B(i-1,j)+B(i,j+1)+B(i,j-1))/(two*D)
           if (lh_soc .ge. zero) then
              ZZ = abs(Z(i,j))
              increment = Zc*Z(i,j)/abs(Z(i,j))
           else
              ZZ = abs(Z(i,j))
              increment = (two*Zc-ZZ)*ran2(idum_run) + ZZ - Zc
           endif
           if ((sigma1 .ne. zero).and.(eps_drive .gt. zero)) then
              Zc_loc = gauss_rand(idum_run,Zc,sigma1/(two*sqrt(two*log(two))))
           else
              Zc_loc = Zc
           endif
           if (ZZ .ge. Zc_loc) then
              if (first_node) then
                 my_rand = ran2(idum_run)
                 first_node = .false.
              endif
              r_dnc = (one-D_nc)*my_rand + D_nc
              if (sigma2 .lt. zero) then
                 r0 = ran2(idum_run)
                 r1 = ran2(idum_run)
                 r2 = ran2(idum_run)
                 r3 = ran2(idum_run)
                 rr = (r0 + r1 + r2 + r3)
                 r0 = two*D*r0*r_dnc/rr
                 r1 = two*D*r1*r_dnc/rr
                 r2 = two*D*r2*r_dnc/rr
                 r3 = two*D*r3*r_dnc/rr

                 ! Avoiding negative energy releases
                 energy_rel2 = increment*( (four*D/s)*B(i,j) - &
                      (two/s)*(r0*B(i+1,j) + r1*B(i-1,j) +r2*B(i,j+1) + r3*B(i,j-1)) - &
                      ((four*(D**2)+r0**2+r1**2+r2**2+r3**2)/s**2)*increment )/e0
                 itran=0
                 do while ((energy_rel2 .lt. zero).and.(itran .lt. 23))
                    itran = itran+1
                    call permute(r0,r1,r2,r3,&
                         r0_tmp,r1_tmp,r2_tmp,r3_tmp,itran)
                    energy_rel2 = increment*( (four*D/s)*B(i,j) - &
                         (two/s)*(r0_tmp*B(i+1,j) + r1_tmp*B(i-1,j) +r2_tmp*B(i,j+1) + r3_tmp*B(i,j-1)) - &
                         ((four*(D**2)+r0_tmp**2+r1_tmp**2+r2_tmp**2+r3_tmp**2)/s**2)*increment )/e0
                 end do
                 if (energy_rel2 .lt. zero) write (*,*) 'OUCH... some negative energy detected...'
                 if (itran .ne. 0) then
                    r0 = r0_tmp
                    r1 = r1_tmp
                    r2 = r2_tmp
                    r3 = r3_tmp
                 endif
              else
                 r0 = r_dnc
                 r1 = r0
                 r2 = r0
                 r3 = r0
              endif
              C(i,j)   = C(i,j)   - two*D*increment/s
              C(i+1,j) = C(i+1,j) + r0*increment/s
              C(i-1,j) = C(i-1,j) + r1*increment/s
              C(i,j+1) = C(i,j+1) + r2*increment/s
              C(i,j-1) = C(i,j-1) + r3*increment/s
              energy_rel = (two*abs(Z(i,j))/Zc-one) - &
                   (r_dnc-one)*D*increment*(B(i,j)-Z(i,j))/(Zc**2) - (r_dnc**2-one)/s
              energy_rel2 = increment*( (four*D/s)*B(i,j) - &
                   (two/s)*(r0*B(i+1,j) + r1*B(i-1,j) +r2*B(i,j+1) + r3*B(i,j-1)) - &
                   ((four*(D**2)+r0**2+r1**2+r2**2+r3**2)/s**2)*increment )/e0
              ! Energy released
              energy = energy + energy_rel
              energy2 = energy2 + energy_rel2
           endif
        enddo
     enddo
     if (verbose) write (*,*) "[DEB] e",energy,first_node

     ! Do the avalanche or update lattice
     if (energy .ne. zero) then
        if (verbose) then
           Write (*,*) "--> Avalanching on it",iter," (with e=",energy,")"
        endif
        old_energy = sum(B*B)/e0
        do i = 2,Nx-1
           do j = 2,Ny-1
              B(i,j) = B(i,j) + C(i,j)
              C(i,j)=zero
           enddo
        enddo
        lattice_energy(iter)  = sum(B*B)/e0
        !energy_released(iter) = energy
        energy_released(iter) = energy2
        if (iter .eq. 1) then
           energy_released2(iter) = zero
        else
           energy_released2(iter) = lattice_energy(iter-1)-lattice_energy(iter)
        endif
        iter=iter+1
     else
        if (eps_drive .lt. zero) then
           !G&V model
           if (verbose) write (*,*) "GV model"
           meanB = sum(B)/(Nx*Ny)
           my_rand=ran2(idum_run)
           deltaB = (my_rand/sigma2)**(one/(-sigma1))-one
           itran = 1
           do while ((deltaB .gt. 1e-1*meanB) .and. (itran .le. itrmax))
              my_rand=ran2(idum_run)
              deltaB = (my_rand/sigma2)**(one/(-sigma1))-one
              itran=itran+1
           enddo
           if (itran .ge. itrmax) write (*,*) 'Warning, a delta B could be too large!'
           my_rand_i=ran2(idum_run)
           i = 2+int((nx-2)*my_rand_i)
           my_rand_j=ran2(idum_run)
           j = 2+int((ny-2)*my_rand_j)
           B(i,j) = B(i,j) + deltaB
           lattice_energy(iter)  = sum(B*B)/e0
           energy_released(iter) = zero
           iter = iter + 1
        else if (eps_drive .eq. zero) then
           ! Lu & Hamilton model
           if (verbose) write (*,*) "LH model"
           my_rand=ran2(idum_run)
           deltaB = (sigma2-sigma1)*my_rand + sigma1
           my_rand_i=ran2(idum_run)
           i = 2+int((nx-2)*my_rand_i)
           my_rand_j=ran2(idum_run)
           j = 2+int((ny-2)*my_rand_j)
           B(i,j) = B(i,j) + deltaB
           lattice_energy(iter)  = sum(B*B)/e0
           energy_released(iter) = zero
           iter = iter + 1
        else
           ! Determinsitic forcing
           if (sigma1 .eq. zero) then
              ! accelerate iteration loop
              !niter_to_do = int(ceiling(log(Zc/maxval(abs(Z)))/log(one+eps_drive)))
              niter_to_do = int(log(Zc/maxval(abs(Z)))/log(one+eps_drive))+1
              if (verbose) then
                 write (*,*) '[DEBUG]: ',log(Zc/maxval(abs(Z))),log(one+eps_drive),log(Zc/maxval(abs(Z)))/log(one+eps_drive)
                 write (*,*) '[DEBUG]: ',Zc,maxval(abs(Z)),niter_to_do
              endif
              if (niter_to_do .eq. 0) niter_to_do = 1 ! just a sanity check
              if (niter_to_do .lt. 0) niter_to_do = Niter-iter+1
              if (niter_to_do .eq. 1) then
                 if ((last_iter_oneiter .eq. iter-1).and.(Niter .ne. 1)) &
                      write (*,*) 'Ouch, doing only 1 iterations, pb is still here!'
                 last_iter_oneiter = iter
              endif
              ! check if we do not do too many iterations
              niter_to_do = min(niter_to_do,Niter-iter+1)
              if (verbose) then
                 write (*,*) "Doing",niter_to_do,"iterations at once..."
                 write (*,*) "We are at iter=",iter,'and we want to do ',Niter,' iterations'
                 write (*,*) "It will bring to ",iter+niter_to_do-1,"its in total."
                 write (*,*) '[DEBUG]: ',log(Zc/maxval(abs(Z))),log(one+eps_drive)
                 write (*,*) '[DEBUG]: ',Zc,maxval(abs(Z))
              endif
              lattice_energy(iter) = (sum(B*B)/e0)*(one+eps_drive)**2
              if (niter_to_do .gt. 1) then
                 do iter_tmp = iter+1,iter+niter_to_do-1
                    lattice_energy(iter_tmp)  = lattice_energy(iter)*((one+eps_drive)**(2*(iter_tmp-iter)))
                 enddo
              endif
              do i = 2,Nx-1
                 do j = 2,Ny-1
                    B(i,j) = B(i,j)*((one+eps_drive)**niter_to_do)
                 enddo
              enddo
              energy_released(iter:iter+niter_to_do-1) = zero
              iter=iter+niter_to_do
           else
              ! Do the forcing
              do i = 2,Nx-1
                 do j = 2,Ny-1
                    B(i,j) = B(i,j)*(one+eps_drive)
                 enddo
              enddo
              energy_released(iter)=zero
              lattice_energy(iter)=sum(B*B)/e0
              iter=iter+1
           endif
        endif
     endif

  enddo

  Deallocate(C)

end subroutine do_avalanche_generic


!!!! current routine !!!
subroutine current(cur, B, i, j, Nx, Ny)!Computes the current approximation at a given (i,j) point on a lattice B(Nx,Ny)
    integer, parameter :: dp = selected_real_kind(15,307) ! double precision
    integer, parameter :: mp = dp
    Integer :: Nx, Ny
    Real(mp) :: B(Nx,Ny)
    Real(mp) :: left, right, up, down, cur
    if (i == 1) then
        left = 0 ! If we are on the edges, the neighbours dont exist so we put them as 0
    else
        left = B(i - 1, j)
    end if
    if (i == Nx) then
        right = 0
    else
        right = B(i + 1, j)
    end if
    if (j == 1) then
        up = 0
    else
        up = B(i, j-1)
    end if
    if (j == Ny) then
        down = 0
    else
        down = B(i, j+1)
    end if
    cur = 4*B(i, j) -left-right-up-down
end subroutine current

!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! SOC_Case = 101 in python !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!
subroutine do_avalanche_generic_dp(Niter, &                 ! INPUT
     Nx,Ny,Binit,Zc,D_nc,eps_drive,sigma1,sigma2,lh_soc,idum_init,verbose,name, &     ! OPTIONAL
     lattice_energy,energy_released,energy_released2,B,Z,last_idum)         ! OUTPUTS
  use permutations
  integer, parameter :: dp = selected_real_kind(15,307) ! double precision
  integer, parameter :: mp =dp
  Integer    :: Nx,Ny
!f2py integer optional,intent(in) :: Nx=16
!f2py integer optional,intent(in) :: Ny=16
  Integer    :: Niter
!f2py intent(in) Niter
  Real(mp) :: Zc,D_nc,eps_drive,sigma1,sigma2
!f2py real(mp) optional, intent(in) :: Zc = 1.0
!f2py real(mp) optional, intent(in) :: D_nc=0.98
!f2py real(mp) optional, intent(in) :: eps_drive=1.e-7
!f2py real(mp) optional, intent(in) :: sigma1=0.0
!f2py real(mp) optional, intent(in) :: sigma2=0.0
  Real(mp) :: lh_soc
!f2py real(mp) optional,intent(in) :: lh_soc=0.0
  Integer  :: idum_init
!f2py integer optional, intent(in) :: idum_init = -67209495
  Real(mp) :: lattice_energy(Niter),energy_released(Niter),energy_released2(Niter)
!f2py intent(out) lattice_energy
!f2py intent(out) energy_released
!f2py intent(out) energy_released2
!f2py depend(Niter) lattice_energy
!f2py depend(Niter) energy_released
!f2py depend(Niter) energy_released2
  Real(mp) :: B(Nx,Ny)
!f2py intent(out) B
!f2py depend(Nx,Ny) B
  Real(mp) :: Z(Nx,Ny)
!f2py intent(out) Z
!f2py depend(Nx,Ny) Z
  Real(mp) :: Binit(Nx,Ny)
!f2py real(mp) optional, intent(in), depend(Nx,Ny) :: Binit = 0.0
  Logical :: verbose
!f2py logical optional, intent(in) :: verbose=0
  Character(len = 10) :: name
!f2py string optional, intent(in) :: name='LH'
  Integer :: last_idum
!f2py intent(out) last_idum

  ! Local variables
  Integer   :: i,j,iter, r
  Real(mp)  :: increment,energy_rel,old_energy,energy2,energy_rel2
  Integer   :: niter_to_do,iter_tmp
  Logical   :: first_node, first_node_rdist
  Real(mp)  :: D,s,e0,my_rand,gauss_rand_dp,my_rand_i,my_rand_j,deltaB,meanB
  !Real(mp)  :: weib_rand_dp, e_LH
  Real(mp)  :: ran1,ZZ,r_dnc,r0,r1,r2,r3,rr,energy,Zc_loc
  Real(mp)  :: r0_tmp,r1_tmp,r2_tmp,r3_tmp
  Real(mp)  :: dnc_loc!,rrtmp
  Integer   :: idum_run,last_iter_oneiter,itran,itrmax
  Real(mp), dimension(:,:), allocatable :: C
  Real(mp)  :: zero,one,two,four,five,ten
  !Real(mp)  :: b_th,d_th
  Real(mp)  :: max_dnc_loc
  Integer   :: idum_rw = -67209495,idum_exp = -6248845
  Real(mp)  :: c_rw,step_rw,exp_rand_dp
  Integer   :: loc_it,it_ch,l_its
  Real(mp)  :: energy0,energy_rel0
  Integer, dimension(1:4) :: dir_number
  Integer, dimension(1:10) :: dir
  Real(mp) :: direction ! direction is for in which direction x is computed in Farhang model
  Real(mp) :: theta, x, cur0, cur1, cur2, cur3, cur_temp  ! Farhang variables
  Real(mp) :: temp_lat(Nx, Ny)
  Integer :: unstable = 0
  Real(mp) :: stats(1:2)

  last_iter_oneiter = -10
  c_rw = 0._mp ! initialize the random walk at zero
  step_rw = 0._mp
  loc_it = 0
  l_its = 1
  it_ch = 0

  max_dnc_loc = 0.
  ! Initialize important variables
  zero = 0._mp
  one  = 1._mp
  two  = 2._mp
  three = 3._mp
  four = 4._mp
  five = 5._mp
  ten = 10._mp
  itrmax=100
  first_node = .true.
  first_node_rdist = .true.

  ! For random numbers
  ! call init_random_seed()
  idum_run = idum_init

  ! Allocate tables
  Allocate(C(Nx,Ny))
  C(:,:) = zero

  D  = two
  s  = two*D+one
  e0 = two*D*(Zc**2)/s

  B(:,:) = Binit
  ! Enforce B = 0 on boundaries
  B(1,:)  = zero
  B(Nx,:) = zero
  B(:,1)  = zero
  B(:,Ny) = zero
  ! Recompute Z
  Z(:,:) = zero
  do i = 2,Nx-1
     do j = 2,Ny-1
        Z(i,j) = B(i,j) - (B(i+1,j)+B(i-1,j)+B(i,j+1)+B(i,j-1))/(two*D)
     end do
  end do
!!$  if (maxval(abs(Z)) .ge. Zc) then
!!$     write (*,*) '*** Warning *** Not starting with a stable state'
!!$  endif

  iter = 1
  ! Main Loop
  stats(1) = zero
  stats(2) = zero
  do while (iter .le. Niter)
      if (name(1:1) .eq. 'F') then
          energy =zero
         energy2=zero
         energy0=zero
         first_node= .true.
         first_node_rdist = .true.
          do i = 1,Nx
               do j = 1,Ny
                  C(i,j)=zero
               enddo
            enddo
         ! Compute curvature
         do i = 2,Nx-1
            do j = 2,Ny-1
               Z(i,j) = B(i,j) - (B(i+1,j)+B(i-1,j)+B(i,j+1)+B(i,j-1))/(two*D)
               ZZ = Z(i,j)
               increment = Zc
               Zc_loc = Zc
               if (ZZ .ge. Zc_loc) then
                  !====Farhang model
                  direction = ran1(idum_run)
                  dir_number(1) = floor(direction*4) + 1 ! chooses a random direction for the optimization in x
                  do r = 2,4
                      if (dir_number(r-1) .eq. 4) then ! the other directions follow
                          dir_number(r) = 1
                      else
                          dir_number(r) = dir_number(r-1)+1
                      end if
                  end do
                  dir = (/i,j, i+1, j, i-1, j, i,j+1, i,j-1/) ! list of the neighbour indexes
                  r0 = ran1(idum_run) ! randomizes NEED TO CHANGE THIS FOR IDUM
                  r1 = ran1(idum_run)
                  r2 = ran1(idum_run)
                  rr = (r0 + r1 + r2)
                  call current(cur0, B, dir(dir_number(1)*2+1), dir(dir_number(1)*2+2), Nx, Ny)
                  call current(cur1, B, dir(dir_number(2)*2+1), dir(dir_number(2)*2+2), Nx, Ny)
                  call current(cur2, B, dir(dir_number(3)*2+1), dir(dir_number(3)*2+2), Nx, Ny)
                  call current(cur3, B, dir(dir_number(4)*2+1), dir(dir_number(4)*2+2), Nx, Ny)
                  theta = -r0*cur0 - r1*cur1 - r2*cur2 + rr*cur3 ! theta is used to find the optimal x
                  x = ((three + two/ten)*(r0**2+r1**2+r2**2)*increment - rr*theta)/& !optimal x
                      (theta + (three + two/ten)*increment*rr)
                  if (name(2:2) == '0') then
                     if (x .lt. zero) then ! if we dont care that x is negative, we have a switch here
                         x = zero
                     end if
                  end if
                  !====redistribution energy computation
                  temp_lat = B! copy B in order to simulate the redistribution and choose if we want to do it
                  temp_lat(i, j) = temp_lat(i, j) - two*D/s*increment ! The redistribution
                  temp_lat(dir(dir_number(1)*2+1), dir(dir_number(1)*2+2)) =&
                          temp_lat(dir(dir_number(1)*2+1), dir(dir_number(1)*2+2))+ two*D/s*r0/(x+rr)*increment
                  temp_lat(dir(dir_number(2)*2+1), dir(dir_number(2)*2+2)) =&
                          temp_lat(dir(dir_number(2)*2+1), dir(dir_number(2)*2+2))+two*D/s*r1/(x+rr)*increment
                  temp_lat(dir(dir_number(3)*2+1), dir(dir_number(3)*2+2)) =&
                          temp_lat(dir(dir_number(3)*2+1), dir(dir_number(3)*2+2))+two*D/s*r2/(x+rr)*increment
                  temp_lat(dir(dir_number(4)*2+1), dir(dir_number(4)*2+2)) =&
                          temp_lat(dir(dir_number(4)*2+1), dir(dir_number(4)*2+2))+two*D/s*x/(x+rr)*increment
                  !====loop to compute the energy of the redistribution
                  energy_rel2 = zero
                  do l=1,5
                      do r=1,5
                          if (i+l-3 .ge. 1) then
                              if (i+l-3 .le. Nx) then
                                  if (j+r-3 .ge. 1) then
                                      if (j+r-3 .le. Nx) then
                                          call current(cur_temp, B, i+l-3, j+r-3, Nx, Ny)
                                          energy_rel2 = energy_rel2 + B(i+l-3, j+r-3)*cur_temp/two
                                          call current(cur_temp, temp_lat, i+l-3, j+r-3, Nx, Ny)
                                          energy_rel2 = energy_rel2 - temp_lat(i+l-3, j+r-3)*cur_temp/two
                                      end if
                                  end if
                              end if
                          end if
                      end do
                  end do
                  itran=0
                  !==== if energy is locally negative, we reconfigure 10 times at most to try and make it
                  do while ((energy_rel2 .lt. zero).and.(itran .lt. 10))
                      if (itran .ge. 9) then
                         if (verbose) then
                             write (*,*) 'exceeded 10 attempts'
                         end if
                      end if
                      itran = itran+1
                      call random_number(direction)
                      dir_number(1) = floor(direction*4) + 1
                      do r = 2,4
                          if (dir_number(r-1) .eq. 4) then
                              dir_number(r) = 1
                          else
                              dir_number(r) = dir_number(r-1)+1
                          end if
                      end do
                      call random_number(r0)
                      call random_number(r1)
                      call random_number(r2)
                      !write(*,*) r0, r1, r2
                      rr = (r0 + r1 + r2)
                      call current(cur0, B, dir(dir_number(1)*2+1), dir(dir_number(1)*2+2), Nx, Ny)
                      call current(cur1, B, dir(dir_number(2)*2+1), dir(dir_number(2)*2+2), Nx, Ny)
                      call current(cur2, B, dir(dir_number(3)*2+1), dir(dir_number(3)*2+2), Nx, Ny)
                      call current(cur3, B, dir(dir_number(4)*2+1), dir(dir_number(4)*2+2), Nx, Ny)
                      theta = -r0*cur0 - r1*cur1 - r2*cur2 + rr*cur3
                      x = ((three+two/ten)*(r1**2+r2**2+r3**2)*Zc - rr*theta)/(theta + (three+two/ten)*Zc*rr)
                      if (name(2:2) == '0') then
                          if (x .lt. zero) then ! if we dont care that x is negative, we have a switch here
                                x = zero
                          end if
                      end if
                      temp_lat = B
                      temp_lat(dir(1), dir(2)) = temp_lat(dir(1), dir(2))- two*D/s*increment
                      temp_lat(dir(dir_number(1)*2+1), dir(dir_number(1)*2+2)) =&
                                temp_lat(dir(dir_number(1)*2+1), dir(dir_number(1)*2+2))+ two*D/s*r0/(x+rr)*increment
                      temp_lat(dir(dir_number(2)*2+1), dir(dir_number(2)*2+2)) =&
                                temp_lat(dir(dir_number(2)*2+1), dir(dir_number(2)*2+2))+two*D/s*r1/(x+rr)*increment
                      temp_lat(dir(dir_number(3)*2+1), dir(dir_number(3)*2+2)) =&
                            temp_lat(dir(dir_number(3)*2+1), dir(dir_number(3)*2+2))+two*D/s*r2/(x+rr)*increment
                      temp_lat(dir(dir_number(4)*2+1), dir(dir_number(4)*2+2)) =&
                            temp_lat(dir(dir_number(4)*2+1), dir(dir_number(4)*2+2))+two*D/s*x/(x+rr)*increment
                      energy_rel2 = zero
                      do l=1,5
                          do r=1,5
                              if (i+l-3 .ge. 1) then
                                  if (i+l-3 .le. Nx) then
                                      if (j+r-3 .ge. 1) then
                                          if (j+r-3 .le. Nx) then
                                              call current(cur_temp, B, i+l-3, j+r-3, Nx, Ny)
                                              energy_rel2 = energy_rel2 + B(i+l-3, j+r-3)*cur_temp/two
                                              call current(cur_temp, temp_lat, i+l-3, j+r-3, Nx, Ny)
                                              energy_rel2 = energy_rel2 - temp_lat(i+l-3, j+r-3)*cur_temp/two
                                          end if
                                      end if
                                  end if
                              end if
                          end do
                      end do
                  end do
                  if (energy_rel2 .lt. zero) then ! If local energy is smaller than 0 after the 20 tries
                     if (verbose) then
                         write (*,*) 'OUCH... some negative energy detected...'
                     end if
                     unstable = 1 ! We choose if we want to redistribute. 1 means that we dont want to
                  end if
                  if (unstable .eq. 0) then ! Farhang redistribution
                      C(i,j)   = C(i,j)   - two*D/s*increment
                      C(dir(dir_number(1)*2+1),dir(dir_number(1)*2+2)) = &
                        C(dir(dir_number(1)*2+1),dir(dir_number(1)*2+2)) + two*D/s*r0/(x+rr)*increment
                      C(dir(dir_number(2)*2+1),dir(dir_number(2)*2+2)) = &
                        C(dir(dir_number(2)*2+1),dir(dir_number(2)*2+2)) + two*D/s*r1/(x+rr)*increment
                      C(dir(dir_number(3)*2+1),dir(dir_number(3)*2+2)) = &
                        C(dir(dir_number(3)*2+1),dir(dir_number(3)*2+2)) + two*D/s*r2/(x+rr)*increment
                      C(dir(dir_number(4)*2+1),dir(dir_number(4)*2+2)) = &
                        C(dir(dir_number(4)*2+1),dir(dir_number(4)*2+2)) + two*D/s*x/(x+rr)*increment
                  else
                      unstable = 0 ! Resets unstable for future iterations
                  end if
               endif
            enddo
         enddo
    !     if (energy2 .lt. zero) then ! This is for the model where we let unstable = 0 but only avalanche when
    !         energy = zero ! the energy of the avalanche is positive
    !     end if
          do i = 1,Nx
               do j = 1,Ny
                  temp_lat(i,j) = B(i,j) + C(i,j)
               enddo
          enddo
          energy2 = zero
          do i = 2,Nx-1
               do j = 2,Ny-1
                  call current(cur_temp, B, i, j, Nx, Ny)
                  energy2 = energy2 + B(i,j)*cur_temp/two
                  call current(cur_temp, temp_lat, i, j, Nx, Ny)
                  energy2 = energy2 - temp_lat(i,j)*cur_temp/two
               enddo
          enddo
          if (energy2 .lt. zero) then
            stats(1) = stats(1) + one
              end if
          stats(2) = stats(2) + one
          write (*,*) stats(1)/stats(2)
         ! Do the avalanche or update lattice
         if (energy2 .gt. zero) then
            if (verbose) then
               Write (*,*) "--> Avalanching on it",iter," (with e=",energy,")"
            endif
            do i = 2,Nx-1
               do j = 2,Ny-1
                  B(i,j) = B(i,j) + C(i,j)
               enddo
            enddo
            lattice_energy(iter) = 0
            do i = 2,Nx-1
               do j = 2,Ny-1
                  call current(cur_temp, B, i, j, Nx, Ny)
                  lattice_energy(iter) = lattice_energy(iter) + B(i,j)*cur_temp/two
               enddo
            enddo
            energy_released(iter) = energy2
            if (iter .eq. 1) then
               energy_released2(iter) = zero
            else
               energy_released2(iter) = lattice_energy(iter-1)-lattice_energy(iter)
            endif

            iter=iter+1
         else
              niter_to_do = int(log(Zc/maxval(abs(Z)))/log(one+eps_drive))+1
!              write (*,*) niter_to_do
              if (verbose) then
                 write (*,*) '[DEBUG]: ',log(Zc/maxval(abs(Z))),log(one+eps_drive),log(Zc/maxval(abs(Z)))/log(one+eps_drive)
                 write (*,*) '[DEBUG]: ',Zc,maxval(abs(Z)),niter_to_do
              endif
              if (niter_to_do .eq. 0) niter_to_do = 1 ! just a sanity check
              if (niter_to_do .lt. 0) then
                  niter_to_do = 1
!                  cur_temp = 0 ! so we use the biggest curv under Zc to drive
!                  do i = 2,Nx-1
!                      do j = 2,Ny-1
!                          if (Z(i,j) .lt. Zc) then
!                              if (Z(i,j) .gt. cur_temp) then
!                                  cur_temp = Z(i,j)
!                              end if
!                          end if
!                      enddo
!                  enddo
!
!                  niter_to_do = int(log(Zc/cur_temp)/log(one+eps_drive))+2
!                  if (niter_to_do > 100) then
!                      niter_to_do = 10
!                  end if
              end if
              niter_to_do = min(niter_to_do,Niter-iter+1)
              if (verbose) then
                 write (*,*) "Doing",niter_to_do,"iterations at once..."
                 write (*,*) "We are at iter=",iter,'and we want to do ',Niter,' iterations'
                 write (*,*) "It will bring to ",iter+niter_to_do-1,"its in total."
                 write (*,*) '[DEBUG]: ',log(Zc/maxval(abs(Z))),log(one+eps_drive)
                 write (*,*) '[DEBUG]: ',Zc,maxval(abs(Z))
              endif
              lattice_energy(iter) = 0
              do i = 2,Nx-1
                  do j = 2,Ny-1
                      call current(cur_temp, B, i, j, Nx, Ny)
                      lattice_energy(iter) = lattice_energy(iter) + B(i,j)*cur_temp/two
                  enddo
              enddo
              lattice_energy(iter) = lattice_energy(iter)*(one+eps_drive)**2
              if (niter_to_do .gt. 1) then
                 do iter_tmp = iter+1,iter+niter_to_do-1
                    lattice_energy(iter_tmp)  = lattice_energy(iter)*((one+eps_drive)**(2*(iter_tmp-iter)))
                 enddo
              endif
              do i = 2,Nx-1
                 do j = 2,Ny-1
                    B(i,j) = B(i,j)*((one+eps_drive)**niter_to_do)
                 enddo
              enddo
              energy_released(iter:iter+niter_to_do-1) = zero
              iter=iter+niter_to_do
         end if
      else
          energy =zero
         energy2=zero
         energy0=zero
         first_node=.true.
         first_node_rdist = .true.
         ! Compute curvature
         do i = 2,Nx-1
            do j = 2,Ny-1
               Z(i,j) = B(i,j) - (B(i+1,j)+B(i-1,j)+B(i,j+1)+B(i,j-1))/(two*D)
               if (lh_soc .ge. zero) then
                  ZZ = abs(Z(i,j))
                  increment = Zc*Z(i,j)/abs(Z(i,j))
               else
                  ZZ = abs(Z(i,j))
                  increment = (two*Zc-ZZ)*ran1(idum_run) + ZZ - Zc
               endif
               if ((sigma1 .gt. zero).and.(eps_drive .gt. zero)) then
                  Zc_loc = gauss_rand_dp(idum_run,Zc,sigma1/(two*sqrt(two*log(two))))
               else
                  Zc_loc = Zc
               endif
               if (ZZ .ge. Zc_loc) then
                  if (first_node) then
                     first_node = .false.
                  endif
    !!$              if (lh_soc .gt. zero) then
    !!$                 rrtmp = sqrt(2.)*sqrt(real(i-Nx/2)**2 + real(j-Ny/2)**2)/Nx
    !!$                 !dnc_loc = D_nc*(rrtmp**lh_soc)
    !!$                 dnc_loc = D_nc*(1.0-rrtmp**lh_soc)
    !!$              else
    !!$                 dnc_loc = D_nc
    !!$              endif
                  dnc_loc = D_nc

                  !if (sigma2 .gt. one) then
                  !   !r_dnc = weib_rand_dp(idum_run,dnc_loc,sigma2,-sigma1)
                  !   ! Determine the new D
                  !   b_th = 5._mp*(B(i+1,j)+B(i-1,j)+B(i,j+1)+B(i,j-1))/(2._mp*Zc)
                  !   d_th = b_th**2 - 4._mp*(5._mp*(two*abs(Z(i,j)/Zc)-one)*(sigma2-1._mp) - b_th - one)
                  !   if (d_th .le. zero) then
                  !      dnc_loc = zero
                  !   else
                  !      dnc_loc = max(zero,(-b_th+sqrt(d_th))/two)
                  !      !write (*,*) dnc_loc,(-b_th+sqrt(d_th))/two
                  !   endif
                  !   my_rand = ran1(idum_run)
                  !   max_dnc_loc = max(max_dnc_loc,dnc_loc)
                  !   !r_dnc = (one-dnc_loc)*my_rand + dnc_loc
                  !   r_dnc = (dnc_loc)*my_rand !+ dnc_loc
                  !   write (*,*) dnc_loc,5._mp*(Z(i,j))/Zc - 9._mp
                  !else
                     my_rand = ran1(idum_run)
                     if (lh_soc .gt. zero) then
                        r_dnc = 2.0*lh_soc*my_rand + dnc_loc - lh_soc
                     else
                        r_dnc = (one-dnc_loc)*my_rand + dnc_loc
                     endif
                     !write (*,*) r_dnc
                  !endif

                  if (sigma2 .lt. zero) then
                     r0 = ran1(idum_run)
                     r1 = ran1(idum_run)
                     r2 = ran1(idum_run)
                     r3 = ran1(idum_run)
                     rr = (r0 + r1 + r2 + r3)
                     r0 = two*D*r0*r_dnc/rr
                     r1 = two*D*r1*r_dnc/rr
                     r2 = two*D*r2*r_dnc/rr
                     r3 = two*D*r3*r_dnc/rr

                     ! Avoiding negative energy releases
                     energy_rel2 = increment*( (four*D/s)*B(i,j) - &
                          (two/s)*(r0*B(i+1,j) + r1*B(i-1,j) +r2*B(i,j+1) + r3*B(i,j-1)) - &
                          ((four*(D**2)+r0**2+r1**2+r2**2+r3**2)/s**2)*increment )/e0
                     itran=0
                     do while ((energy_rel2 .lt. zero).and.(itran .lt. 23))
                        itran = itran+1
                        call permute_dp(r0,r1,r2,r3,&
                             r0_tmp,r1_tmp,r2_tmp,r3_tmp,itran)
                        energy_rel2 = increment*( (four*D/s)*B(i,j) - &
                             (two/s)*(r0_tmp*B(i+1,j) + r1_tmp*B(i-1,j) +r2_tmp*B(i,j+1) + r3_tmp*B(i,j-1)) - &
                             ((four*(D**2)+r0_tmp**2+r1_tmp**2+r2_tmp**2+r3_tmp**2)/s**2)*increment )/e0
                     end do
                     if (energy_rel2 .lt. zero) write (*,*) 'OUCH... some negative energy detected...'
                     if (itran .ne. 0) then
                        r0 = r0_tmp
                        r1 = r1_tmp
                        r2 = r2_tmp
                        r3 = r3_tmp
                     endif
                  else
                     r0 = r_dnc
                     r1 = r0
                     r2 = r0
                     r3 = r0
                     !ra = 1.0
                  endif

                  energy_rel0 = (two*abs(Z(i,j))/Zc-one)

                  energy_rel = (two*abs(Z(i,j))/Zc-one) - &
                       (r_dnc-one)*D*increment*(B(i,j)-Z(i,j))/(Zc**2) - (r_dnc**2-one)/s
                  if (sigma2 .gt. zero) then
                     energy_rel2 = increment*( (four*r0*D/s)*B(i,j) - &
                          (two/s)*(B(i+1,j) + B(i-1,j) + B(i,j+1) + B(i,j-1)) - &
                          ((four*((r0*D)**2)+four)/s**2)*increment )/e0
                  else
                     energy_rel2 = increment*( (four*D/s)*B(i,j) - &
                          (two/s)*(r0*B(i+1,j) + r1*B(i-1,j) +r2*B(i,j+1) + r3*B(i,j-1)) - &
                          ((four*(D**2)+r0**2+r1**2+r2**2+r3**2)/s**2)*increment )/e0
                  endif

    !!$              e_LH = two*abs(Z(i,j))/Zc-one
    !!$              !write (*,*) energy_rel2,e_LH
    !!$              do while (energy_rel2 .ge. 100*e_LH)
    !!$                 ! try to find another one
    !!$                 if (sigma2 .gt. zero) then
    !!$                    r_dnc = weib_rand_dp(idum_run,dnc_loc,sigma2,-sigma1)
    !!$                 else
    !!$                    my_rand = ran1(idum_run)
    !!$                    if (lh_soc .gt. zero) then
    !!$                       r_dnc = 2.0*lh_soc*my_rand + dnc_loc - lh_soc
    !!$                    else
    !!$                       r_dnc = (one-dnc_loc)*my_rand + dnc_loc
    !!$                    endif
    !!$                    r0 = r_dnc
    !!$                    r1 = r0
    !!$                    r2 = r0
    !!$                    r3 = r0
    !!$                 endif
    !!$                 energy_rel2 = increment*( (four*D/s)*B(i,j) - &
    !!$                      (two/s)*(r0*B(i+1,j) + r1*B(i-1,j) +r2*B(i,j+1) + r3*B(i,j-1)) - &
    !!$                      ((four*(D**2)+r0**2+r1**2+r2**2+r3**2)/s**2)*increment )/e0
    !!$                 e_LH = two*abs(Z(i,j))/Zc-one
    !!$                 !write (*,*) energy_rel2,e_LH
    !!$              end do
                  ! end to be erased afterwards

                  if (sigma2 .gt. zero) then
                     C(i,j)   = C(i,j)   - two*D*r0*increment/s
                     C(i+1,j) = C(i+1,j) + increment/s
                     C(i-1,j) = C(i-1,j) + increment/s
                     C(i,j+1) = C(i,j+1) + increment/s
                     C(i,j-1) = C(i,j-1) + increment/s
                  else
                     C(i,j)   = C(i,j)   - two*D*increment/s
                     C(i+1,j) = C(i+1,j) + r0*increment/s
                     C(i-1,j) = C(i-1,j) + r1*increment/s
                     C(i,j+1) = C(i,j+1) + r2*increment/s
                     C(i,j-1) = C(i,j-1) + r3*increment/s
                  endif

                  ! Energy released
                  energy0 = energy0 + energy_rel0
                  energy = energy + energy_rel
                  energy2 = energy2 + energy_rel2
               endif
            enddo
         enddo
         !if (verbose) write (*,*) "[DEB] e",energy,first_node

         ! Do the avalanche or update lattice
         if (energy .ne. zero) then
            if (verbose) then
               Write (*,*) "--> Avalanching on it",iter," (with e=",energy,")"
            endif
            old_energy = sum(B*B)/e0
            do i = 2,Nx-1
               do j = 2,Ny-1
                  B(i,j) = B(i,j) + C(i,j)
                  C(i,j)=zero
               enddo
            enddo
            lattice_energy(iter)  = sum(B*B)/e0
            !energy_released(iter) = energy
            energy_released(iter) = energy2
            if (iter .eq. 1) then
               energy_released2(iter) = zero
            else
               energy_released2(iter) = lattice_energy(iter-1)-lattice_energy(iter)
            endif

            !!! TO BE ERASED
            !energy_released(iter) = energy0
            !!! END

            iter=iter+1
    !!$        ! Try a continuous forcing
    !!$        if (sigma1 .lt. zero) then
    !!$           do i = 2,Nx-1
    !!$              do j = 2,Ny-1
    !!$                 B(i,j) = B(i,j)*(one+eps_drive)
    !!$              enddo
    !!$           enddo
    !!$        endif
         else
            if (eps_drive .lt. zero) then
               !G&V model
               if (verbose) write (*,*) "GV model"
               meanB = sum(B)/(Nx*Ny)
               my_rand=ran1(idum_run)
               deltaB = (my_rand/sigma2)**(one/(-sigma1))-one
               itran = 1
               do while ((deltaB .gt. 1e-1*meanB) .and. (itran .le. itrmax))
                  my_rand=ran1(idum_run)
                  deltaB = (my_rand/sigma2)**(one/(-sigma1))-one
                  itran=itran+1
               enddo
               if (itran .ge. itrmax) write (*,*) 'Warning, a delta B could be too large!'
               my_rand_i=ran1(idum_run)
               i = 2+int((nx-2)*my_rand_i)
               my_rand_j=ran1(idum_run)
               j = 2+int((ny-2)*my_rand_j)
               B(i,j) = B(i,j) + deltaB
               lattice_energy(iter)  = sum(B*B)/e0
               energy_released(iter) = zero
               iter = iter + 1
            else if (eps_drive .eq. zero) then
               ! Lu & Hamilton model
               if (verbose) write (*,*) "LH model"
               my_rand=ran1(idum_run)
               deltaB = (sigma2-sigma1)*my_rand + sigma1
               my_rand_i=ran1(idum_run)
               i = 2+int((nx-2)*my_rand_i)
               my_rand_j=ran1(idum_run)
               j = 2+int((ny-2)*my_rand_j)
               B(i,j) = B(i,j) + deltaB
               lattice_energy(iter)  = sum(B*B)/e0
               energy_released(iter) = zero
               iter = iter + 1
            else
               ! Determinsitic forcing
               if (sigma1 .eq. zero) then
                  ! accelerate iteration loop
                  !niter_to_do = int(ceiling(log(Zc/maxval(abs(Z)))/log(one+eps_drive)))
                  niter_to_do = int(log(Zc/maxval(abs(Z)))/log(one+eps_drive))+1
                  if (verbose) then
                     write (*,*) '[DEBUG]: ',log(Zc/maxval(abs(Z))),log(one+eps_drive),log(Zc/maxval(abs(Z)))/log(one+eps_drive)
                     write (*,*) '[DEBUG]: ',Zc,maxval(abs(Z)),niter_to_do
                  endif
                  if (niter_to_do .eq. 0) niter_to_do = 1 ! just a sanity check
                  if (niter_to_do .lt. 0) niter_to_do = Niter-iter+1
                  if (niter_to_do .eq. 1) then
                     if (last_iter_oneiter .eq. iter-1) then
                        write (*,*) 'Ouch, doing only 1 iterations, pb is still here!'
                     endif
                     last_iter_oneiter = iter
                  endif
                  ! check if we do not do too many iterations
                  niter_to_do = min(niter_to_do,Niter-iter+1)
                  if (verbose) then
                     write (*,*) "Doing",niter_to_do,"iterations at once..."
                     write (*,*) "We are at iter=",iter,'and we want to do ',Niter,' iterations'
                     write (*,*) "It will bring to ",iter+niter_to_do-1,"its in total."
                     write (*,*) '[DEBUG]: ',log(Zc/maxval(abs(Z))),log(one+eps_drive)
                     write (*,*) '[DEBUG]: ',Zc,maxval(abs(Z))
                  endif
                  lattice_energy(iter) = (sum(B*B)/e0)*(one+eps_drive)**2
                  if (niter_to_do .gt. 1) then
                     do iter_tmp = iter+1,iter+niter_to_do-1
                        lattice_energy(iter_tmp)  = lattice_energy(iter)*((one+eps_drive)**(2*(iter_tmp-iter)))
                     enddo
                  endif
                  do i = 2,Nx-1
                     do j = 2,Ny-1
                        B(i,j) = B(i,j)*((one+eps_drive)**niter_to_do)
                     enddo
                  enddo
                  energy_released(iter:iter+niter_to_do-1) = zero
                  iter=iter+niter_to_do
               else
                  niter_to_do = int(log(Zc/maxval(abs(Z)))/log(one+eps_drive))+1
                  if (niter_to_do .eq. 0) niter_to_do = 1 ! just a sanity check
                  if (niter_to_do .lt. 0) niter_to_do = Niter-iter+1
                  niter_to_do = min(niter_to_do,Niter-iter+1)
                  ! Do the forcing
                  !if (sigma1 .lt. zero) then
                  !   step_rw = ran2(idum_rw)
                  !   if (step_rw .gt. 0.5) then
                  !      c_rw = c_rw - sigma1
                  !   else
                  !      c_rw = c_rw + sigma1
                  !   endif
                  !else
                  !   c_rw = 1.0
                  !endif
                  if (sigma1 .lt. zero) then
                     loc_it = iter - it_ch
                     if ((c_rw .eq. 0._mp) .or. (loc_it .ge. l_its)) then
                        l_its = int(gauss_rand_dp(idum_exp,sigma2,sigma2/2.))
                        c_rw = exp_rand_dp(idum_rw,-sigma1)
                        do while (c_rw/eps_drive .gt. (niter_to_do+1))
                           c_rw = exp_rand_dp(idum_rw,-sigma1)
                        end do
                        it_ch = iter+1
                     endif
                  else
                     c_rw = eps_drive
                  endif
                  step_rw = max(step_rw,c_rw)
                  do i = 2,Nx-1
                     do j = 2,Ny-1
                        !B(i,j) = B(i,j)*(one+abs(c_rw)*eps_drive)
                        B(i,j) = B(i,j)*(one+c_rw)
                     enddo
                  enddo
                  energy_released(iter)=zero
                  lattice_energy(iter)=sum(B*B)/e0
                  iter=iter+1
               endif
            endif
            endif
      end if
  enddo

  !write (*,*) 'The max drive was',step_rw
  last_idum = idum_run

!!$  write (*,*) 'Max dnc_loc = ',max_dnc_loc,' for sigma2=',sigma2

end subroutine do_avalanche_generic_dp

!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! SOC_Case = 102 in python !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!
subroutine do_avalanche_generic_ho(Niter, &                 ! INPUT
     Nx,Ny,Binit,Zc,D_nc,eps_drive,sigma1,sigma2,lh_soc,idum_init,verbose, &     ! OPTIONAL
     lattice_energy,energy_released,energy_released2,B,Z)         ! OUTPUTS
  use permutations
  integer, parameter :: dp = selected_real_kind(15,307) ! double precision
  integer, parameter :: mp =dp
  Integer    :: Nx,Ny
!f2py integer optional,intent(in) :: Nx=16
!f2py integer optional,intent(in) :: Ny=16
  Integer    :: Niter
!f2py intent(in) Niter
  Real(mp) :: Zc,D_nc,eps_drive,sigma1,sigma2
!f2py real(mp) optional, intent(in) :: Zc = 1.0
!f2py real(mp) optional, intent(in) :: D_nc=0.98
!f2py real(mp) optional, intent(in) :: eps_drive=1.e-7
!f2py real(mp) optional, intent(in) :: sigma1=0.0
!f2py real(mp) optional, intent(in) :: sigma2=0.0
  Real(mp) :: lh_soc
!f2py real(mp) optional,intent(in) :: lh_soc=0.0
  Integer :: idum_init
!f2py integer optional, intent(in) :: idum_init = -67209495
  Real(mp) :: lattice_energy(Niter),energy_released(Niter),energy_released2(Niter)
!f2py intent(out) lattice_energy
!f2py intent(out) energy_released
!f2py intent(out) energy_released2
!f2py depend(Niter) lattice_energy
!f2py depend(Niter) energy_released
!f2py depend(Niter) energy_released2
  Real(mp) :: B(Nx,Ny)
!f2py intent(out) B
!f2py depend(Nx,Ny) B
  Real(mp) :: Z(Nx,Ny)
!f2py intent(out) Z
!f2py depend(Nx,Ny) Z
  Real(mp) :: Binit(Nx,Ny)
!f2py real(mp) optional, intent(in), depend(Nx,Ny) :: Binit = 0.0
  Logical :: verbose
!f2py logical optional, intent(in) :: verbose=0

  ! Local variables
  Integer   :: i,j,iter
  Real(mp)  :: increment,energy_rel,old_energy,energy2,energy_rel2
  Integer   :: niter_to_do,iter_tmp
  Logical   :: first_node, first_node_rdist
  Real(mp)  :: D,s,e0,my_rand,gauss_rand_dp,my_rand_i,my_rand_j,deltaB,meanB
  Real(mp)  :: ran1,ZZ,r_dnc,r0,r1,r2,r3,rr,energy,Zc_loc
  Real(mp)  :: r0_tmp,r1_tmp,r2_tmp,r3_tmp
  Integer   :: idum_run,last_iter_oneiter,itran,itrmax
  Real(mp), dimension(:,:), allocatable :: C
  Real(mp)  :: zero,one,two,four

  last_iter_oneiter = -10

  ! Initialize important variables
  zero = 0._mp
  one  = 1._mp
  two  = 2._mp
  four = 4._mp
  itrmax=100
  first_node = .true.
  first_node_rdist = .true.

  ! For random numbers
  ! call init_random_seed()
  idum_run = idum_init

  ! Allocate tables
  Allocate(C(Nx,Ny))
  C(:,:) = zero

  D  = two
  s  = two*D+one
  e0 = two*D*(Zc**2)/s

  B(:,:) = Binit
  ! Enforce B = 0 on boundaries
  B(1,:)  = zero
  B(Nx,:) = zero
  B(:,1)  = zero
  B(:,Ny) = zero
  B(2,:)  = zero
  B(Nx-1,:) = zero
  B(:,2)  = zero
  B(:,Ny-1) = zero
  ! Recompute Z
  Z(:,:) = zero
  do i = 3,Nx-2
     do j = 3,Ny-2
        !Z(i,j) = B(i,j) - (B(i+1,j)+B(i-1,j)+B(i,j+1)+B(i,j-1))/(two*D)
        Z(i,j) = B(i,j) - ( (4._mp/3._mp)*(B(i+1,j)+B(i-1,j)+B(i,j+1)+B(i,j-1)) - &
             (1._mp/12._mp)*(B(i+2,j)+B(i-2,j)+B(i,j+2)+B(i,j-2)) )/5._mp
     end do
  end do
!!$  if (maxval(abs(Z)) .ge. Zc) then
!!$     write (*,*) '*** Warning *** Not starting with a stable state'
!!$  endif

  iter = 1
  ! Main Loop
  do while (iter .le. Niter)
     energy =zero
     energy2=zero
     first_node=.true.
     first_node_rdist = .true.
     ! Compute curvature
     do i = 3,Nx-2
        do j = 3,Ny-2
           !Z(i,j) = B(i,j) - (B(i+1,j)+B(i-1,j)+B(i,j+1)+B(i,j-1))/(two*D)
           Z(i,j) = B(i,j) - ( (4._mp/3._mp)*(B(i+1,j)+B(i-1,j)+B(i,j+1)+B(i,j-1)) - &
                (1._mp/12._mp)*(B(i+2,j)+B(i-2,j)+B(i,j+2)+B(i,j-2)) )/5._mp
           if (lh_soc .ge. zero) then
              ZZ = abs(Z(i,j))
              increment = Zc*Z(i,j)/abs(Z(i,j))
           else
              ZZ = abs(Z(i,j))
              increment = (two*Zc-ZZ)*ran1(idum_run) + ZZ - Zc
           endif
           if ((sigma1 .ne. zero).and.(eps_drive .gt. zero)) then
              Zc_loc = gauss_rand_dp(idum_run,Zc,sigma1/(two*sqrt(two*log(two))))
           else
              Zc_loc = Zc
           endif
           if (ZZ .ge. Zc_loc) then
              if (first_node) then
                 my_rand = ran1(idum_run)
                 first_node = .false.
              endif
              r_dnc = (one-D_nc)*my_rand + D_nc
              if (sigma2 .lt. zero) then
                 r0 = ran1(idum_run)
                 r1 = ran1(idum_run)
                 r2 = ran1(idum_run)
                 r3 = ran1(idum_run)
                 rr = (r0 + r1 + r2 + r3)
                 r0 = two*D*r0*r_dnc/rr
                 r1 = two*D*r1*r_dnc/rr
                 r2 = two*D*r2*r_dnc/rr
                 r3 = two*D*r3*r_dnc/rr

                 ! Avoiding negative energy releases
                 energy_rel2 = increment*( (four*D/s)*B(i,j) - &
                      (two/s)*(r0*B(i+1,j) + r1*B(i-1,j) +r2*B(i,j+1) + r3*B(i,j-1)) - &
                      ((four*(D**2)+r0**2+r1**2+r2**2+r3**2)/s**2)*increment )/e0
                 itran=0
                 do while ((energy_rel2 .lt. zero).and.(itran .lt. 23))
                    itran = itran+1
                    call permute_dp(r0,r1,r2,r3,&
                         r0_tmp,r1_tmp,r2_tmp,r3_tmp,itran)
                    energy_rel2 = increment*( (four*D/s)*B(i,j) - &
                         (two/s)*(r0_tmp*B(i+1,j) + r1_tmp*B(i-1,j) +r2_tmp*B(i,j+1) + r3_tmp*B(i,j-1)) - &
                         ((four*(D**2)+r0_tmp**2+r1_tmp**2+r2_tmp**2+r3_tmp**2)/s**2)*increment )/e0
                 end do
                 if (energy_rel2 .lt. zero) write (*,*) 'OUCH... some negative energy detected...'
                 if (itran .ne. 0) then
                    r0 = r0_tmp
                    r1 = r1_tmp
                    r2 = r2_tmp
                    r3 = r3_tmp
                 endif
              else
                 r0 = r_dnc
                 r1 = r0
                 r2 = r0
                 r3 = r0
                 !ra = 1.0
              endif
              C(i,j)   = C(i,j)   - two*D*increment/s
              C(i+1,j) = C(i+1,j) + r0*increment/s
              C(i-1,j) = C(i-1,j) + r1*increment/s
              C(i,j+1) = C(i,j+1) + r2*increment/s
              C(i,j-1) = C(i,j-1) + r3*increment/s
              energy_rel = (two*abs(Z(i,j))/Zc-one) - &
                   (r_dnc-one)*D*increment*(B(i,j)-Z(i,j))/(Zc**2) - (r_dnc**2-one)/s
              energy_rel2 = increment*( (four*D/s)*B(i,j) - &
                   (two/s)*(r0*B(i+1,j) + r1*B(i-1,j) +r2*B(i,j+1) + r3*B(i,j-1)) - &
                   ((four*(D**2)+r0**2+r1**2+r2**2+r3**2)/s**2)*increment )/e0
              ! Energy released
              energy = energy + energy_rel
              energy2 = energy2 + energy_rel2
           endif
        enddo
     enddo
     if (verbose) write (*,*) "[DEB] e",energy,first_node

     ! Do the avalanche or update lattice
     if (energy .ne. zero) then
        if (verbose) then
           Write (*,*) "--> Avalanching on it",iter," (with e=",energy,")"
        endif
        old_energy = sum(B*B)/e0
        do i = 3,Nx-2
           do j = 3,Ny-2
              B(i,j) = B(i,j) + C(i,j)
              C(i,j)=zero
           enddo
        enddo
        lattice_energy(iter)  = sum(B*B)/e0
        !energy_released(iter) = energy
        energy_released(iter) = energy2
        if (iter .eq. 1) then
           energy_released2(iter) = zero
        else
           energy_released2(iter) = lattice_energy(iter-1)-lattice_energy(iter)
        endif
        iter=iter+1
     else
        if (eps_drive .lt. zero) then
           !G&V model
           if (verbose) write (*,*) "GV model"
           meanB = sum(B)/(Nx*Ny)
           my_rand=ran1(idum_run)
           deltaB = (my_rand/sigma2)**(one/(-sigma1))-one
           itran = 1
           do while ((deltaB .gt. 1e-1*meanB) .and. (itran .le. itrmax))
              my_rand=ran1(idum_run)
              deltaB = (my_rand/sigma2)**(one/(-sigma1))-one
              itran=itran+1
           enddo
           if (itran .ge. itrmax) write (*,*) 'Warning, a delta B could be too large!'
           my_rand_i=ran1(idum_run)
           i = 3+int((nx-4)*my_rand_i)
           my_rand_j=ran1(idum_run)
           j = 3+int((ny-4)*my_rand_j)
           B(i,j) = B(i,j) + deltaB
           lattice_energy(iter)  = sum(B*B)/e0
           energy_released(iter) = zero
           iter = iter + 1
        else if (eps_drive .eq. zero) then
           ! Lu & Hamilton model
           if (verbose) write (*,*) "LH model"
           my_rand=ran1(idum_run)
           deltaB = (sigma2-sigma1)*my_rand + sigma1
           my_rand_i=ran1(idum_run)
           i = 3+int((nx-4)*my_rand_i)
           my_rand_j=ran1(idum_run)
           j = 3+int((ny-4)*my_rand_j)
           B(i,j) = B(i,j) + deltaB
           lattice_energy(iter)  = sum(B*B)/e0
           energy_released(iter) = zero
           iter = iter + 1
        else
           ! Determinsitic forcing
           if (sigma1 .eq. zero) then
              ! accelerate iteration loop
              !niter_to_do = int(ceiling(log(Zc/maxval(abs(Z)))/log(one+eps_drive)))
              niter_to_do = int(log(Zc/maxval(abs(Z)))/log(one+eps_drive))+1
              if (verbose) then
                 write (*,*) '[DEBUG]: ',log(Zc/maxval(abs(Z))),log(one+eps_drive),log(Zc/maxval(abs(Z)))/log(one+eps_drive)
                 write (*,*) '[DEBUG]: ',Zc,maxval(abs(Z)),niter_to_do
              endif
              if (niter_to_do .eq. 0) niter_to_do = 1 ! just a sanity check
              if (niter_to_do .lt. 0) niter_to_do = Niter-iter+1
              if (niter_to_do .eq. 1) then
                 if (last_iter_oneiter .eq. iter-1) then
                    write (*,*) 'Ouch, doing only 1 iterations, pb is still here!'
                 endif
                 last_iter_oneiter = iter
              endif
              ! check if we do not do too many iterations
              niter_to_do = min(niter_to_do,Niter-iter+1)
              if (verbose) then
                 write (*,*) "Doing",niter_to_do,"iterations at once..."
                 write (*,*) "We are at iter=",iter,'and we want to do ',Niter,' iterations'
                 write (*,*) "It will bring to ",iter+niter_to_do-1,"its in total."
                 write (*,*) '[DEBUG]: ',log(Zc/maxval(abs(Z))),log(one+eps_drive)
                 write (*,*) '[DEBUG]: ',Zc,maxval(abs(Z))
              endif
              lattice_energy(iter) = (sum(B*B)/e0)*(one+eps_drive)**2
              if (niter_to_do .gt. 1) then
                 do iter_tmp = iter+1,iter+niter_to_do-1
                    lattice_energy(iter_tmp)  = lattice_energy(iter)*((one+eps_drive)**(2*(iter_tmp-iter)))
                 enddo
              endif
              do i = 3,Nx-2
                 do j = 3,Ny-2
                    B(i,j) = B(i,j)*((one+eps_drive)**niter_to_do)
                 enddo
              enddo
              energy_released(iter:iter+niter_to_do-1) = zero
              iter=iter+niter_to_do
           else
              ! Do the forcing
              do i = 3,Nx-2
                 do j = 3,Ny-2
                    B(i,j) = B(i,j)*(one+eps_drive)
                 enddo
              enddo
              energy_released(iter)=zero
              lattice_energy(iter)=sum(B*B)/e0
              iter=iter+1
           endif
        endif
     endif

  enddo

end subroutine do_avalanche_generic_ho

!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! SOC_Case = 103 in python !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!
subroutine do_avalanche_generic_NCWEIRD(Niter, &                 ! INPUT
     Nx,Ny,Binit,Zc,D_nc,eps_drive,sigma1,sigma2,lh_soc,idum_init,verbose, &     ! OPTIONAL
     lattice_energy,energy_released,energy_released2,B,Z)         ! OUTPUTS
  use permutations
  integer, parameter :: dp = selected_real_kind(15,307) ! double precision
  integer, parameter :: mp =dp
  Integer    :: Nx,Ny
!f2py integer optional,intent(in) :: Nx=16
!f2py integer optional,intent(in) :: Ny=16
  Integer    :: Niter
!f2py intent(in) Niter
  Real(mp) :: Zc,D_nc,eps_drive,sigma1,sigma2
!f2py real(mp) optional, intent(in) :: Zc = 1.0
!f2py real(mp) optional, intent(in) :: D_nc=0.98
!f2py real(mp) optional, intent(in) :: eps_drive=1.e-7
!f2py real(mp) optional, intent(in) :: sigma1=0.0
!f2py real(mp) optional, intent(in) :: sigma2=0.0
  Real(mp) :: lh_soc
!f2py real(mp) optional,intent(in) :: lh_soc=0.0
  Integer :: idum_init
!f2py integer optional, intent(in) :: idum_init = -67209495
  Real(mp) :: lattice_energy(Niter),energy_released(Niter),energy_released2(Niter)
!f2py intent(out) lattice_energy
!f2py intent(out) energy_released
!f2py intent(out) energy_released2
!f2py depend(Niter) lattice_energy
!f2py depend(Niter) energy_released
!f2py depend(Niter) energy_released2
  Real(mp) :: B(Nx,Ny)
!f2py intent(out) B
!f2py depend(Nx,Ny) B
  Real(mp) :: Z(Nx,Ny)
!f2py intent(out) Z
!f2py depend(Nx,Ny) Z
  Real(mp) :: Binit(Nx,Ny)
!f2py real(mp) optional, intent(in), depend(Nx,Ny) :: Binit = 0.0
  Logical :: verbose
!f2py logical optional, intent(in) :: verbose=0

  ! Local variables
  Integer   :: i,j,iter
  Real(mp)  :: increment,energy_rel,old_energy,energy2,energy_rel2
  Integer   :: niter_to_do,iter_tmp
  Logical   :: first_node, first_node_rdist
  Real(mp)  :: D,s,e0,my_rand,gauss_rand_dp,my_rand_i,my_rand_j,deltaB,meanB,weib_rand_dp
  Real(mp)  :: ran1,ZZ,r_dnc,r0,r1,r2,r3,rr,energy,Zc_loc
  Real(mp)  :: r0_tmp,r1_tmp,r2_tmp,r3_tmp
  Integer   :: idum_run,last_iter_oneiter,itran,itrmax
  Real(mp), dimension(:,:), allocatable :: C
  Real(mp)  :: zero,one,two,four

  last_iter_oneiter = -10

  ! Initialize important variables
  zero = 0._mp
  one  = 1._mp
  two  = 2._mp
  four = 4._mp
  itrmax=100
  first_node = .true.
  first_node_rdist = .true.

  ! For random numbers
  ! call init_random_seed()
  idum_run = idum_init

  ! Allocate tables
  Allocate(C(Nx,Ny))
  C(:,:) = zero

  D  = two
  s  = two*D+one
  e0 = two*D*(Zc**2)/s

  B(:,:) = Binit
  ! Enforce B = 0 on boundaries
  B(1,:)  = zero
  B(Nx,:) = zero
  B(:,1)  = zero
  B(:,Ny) = zero
  ! Recompute Z
  Z(:,:) = zero
  do i = 2,Nx-1
     do j = 2,Ny-1
        Z(i,j) = B(i,j) - (B(i+1,j)+B(i-1,j)+B(i,j+1)+B(i,j-1))/(two*D)
     end do
  end do
!!$  if (maxval(abs(Z)) .ge. Zc) then
!!$     write (*,*) '*** Warning *** Not starting with a stable state'
!!$  endif

  iter = 1
  ! Main Loop
  do while (iter .le. Niter)
     energy =zero
     energy2=zero
     first_node=.true.
     first_node_rdist = .true.
     ! Compute curvature
     do i = 2,Nx-1
        do j = 2,Ny-1
           Z(i,j) = B(i,j) - (B(i+1,j)+B(i-1,j)+B(i,j+1)+B(i,j-1))/(two*D)
           if (lh_soc .ge. zero) then
              ZZ = abs(Z(i,j))
              increment = Zc*Z(i,j)/abs(Z(i,j))
           else
              ZZ = abs(Z(i,j))
              increment = (two*Zc-ZZ)*ran1(idum_run) + ZZ - Zc
           endif
           if ((sigma1 .ne. zero).and.(eps_drive .gt. zero)) then
              Zc_loc = gauss_rand_dp(idum_run,Zc,sigma1/(two*sqrt(two*log(two))))
           else
              Zc_loc = Zc
           endif
           if (ZZ .ge. Zc_loc) then
              if (first_node) then
                 !my_rand = ran1(idum_run)
                 !my_rand = gauss_rand_dp(idum_run,D_nc,0.1/(two*sqrt(two*log(two))))
                 my_rand = weib_rand_dp(idum_run,0.9,D_nc)
                 first_node = .false.
              endif
              !r_dnc = (one-D_nc)*my_rand + D_nc
              !r_dnc = (1.5*D_nc)*my_rand + one - 0.5*D_nc
              r_dnc = max(my_rand,D_nc)
              r_dnc = min(r_dnc,one)
              if (sigma2 .lt. zero) then
                 r0 = ran1(idum_run)
                 r1 = ran1(idum_run)
                 r2 = ran1(idum_run)
                 r3 = ran1(idum_run)
                 rr = (r0 + r1 + r2 + r3)
                 r0 = two*D*r0*r_dnc/rr
                 r1 = two*D*r1*r_dnc/rr
                 r2 = two*D*r2*r_dnc/rr
                 r3 = two*D*r3*r_dnc/rr

                 ! Avoiding negative energy releases
                 energy_rel2 = increment*( (four*D/s)*B(i,j) - &
                      (two/s)*(r0*B(i+1,j) + r1*B(i-1,j) +r2*B(i,j+1) + r3*B(i,j-1)) - &
                      ((four*(D**2)+r0**2+r1**2+r2**2+r3**2)/s**2)*increment )/e0
                 itran=0
                 do while ((energy_rel2 .lt. zero).and.(itran .lt. 23))
                    itran = itran+1
                    call permute_dp(r0,r1,r2,r3,&
                         r0_tmp,r1_tmp,r2_tmp,r3_tmp,itran)
                    energy_rel2 = increment*( (four*D/s)*B(i,j) - &
                         (two/s)*(r0_tmp*B(i+1,j) + r1_tmp*B(i-1,j) +r2_tmp*B(i,j+1) + r3_tmp*B(i,j-1)) - &
                         ((four*(D**2)+r0_tmp**2+r1_tmp**2+r2_tmp**2+r3_tmp**2)/s**2)*increment )/e0
                 end do
                 if (energy_rel2 .lt. zero) write (*,*) 'OUCH... some negative energy detected...'
                 if (itran .ne. 0) then
                    r0 = r0_tmp
                    r1 = r1_tmp
                    r2 = r2_tmp
                    r3 = r3_tmp
                 endif
              else
                 r0 = r_dnc
                 r1 = r0
                 r2 = r0
                 r3 = r0
                 !ra = 1.0
              endif
              C(i,j)   = C(i,j)   - two*D*increment/s
              C(i+1,j) = C(i+1,j) + r0*increment/s
              C(i-1,j) = C(i-1,j) + r1*increment/s
              C(i,j+1) = C(i,j+1) + r2*increment/s
              C(i,j-1) = C(i,j-1) + r3*increment/s
              energy_rel = (two*abs(Z(i,j))/Zc-one) - &
                   (r_dnc-one)*D*increment*(B(i,j)-Z(i,j))/(Zc**2) - (r_dnc**2-one)/s
              energy_rel2 = increment*( (four*D/s)*B(i,j) - &
                   (two/s)*(r0*B(i+1,j) + r1*B(i-1,j) +r2*B(i,j+1) + r3*B(i,j-1)) - &
                   ((four*(D**2)+r0**2+r1**2+r2**2+r3**2)/s**2)*increment )/e0
              ! Energy released
              energy = energy + energy_rel
              energy2 = energy2 + energy_rel2
           endif
        enddo
     enddo
     if (verbose) write (*,*) "[DEB] e",energy,first_node

     ! Do the avalanche or update lattice
     if (energy .ne. zero) then
        if (verbose) then
           Write (*,*) "--> Avalanching on it",iter," (with e=",energy,")"
        endif
        old_energy = sum(B*B)/e0
        do i = 2,Nx-1
           do j = 2,Ny-1
              B(i,j) = B(i,j) + C(i,j)
              C(i,j)=zero
           enddo
        enddo
        lattice_energy(iter)  = sum(B*B)/e0
        !energy_released(iter) = energy
        energy_released(iter) = energy2
        if (iter .eq. 1) then
           energy_released2(iter) = zero
        else
           energy_released2(iter) = lattice_energy(iter-1)-lattice_energy(iter)
        endif
        iter=iter+1
     else
        if (eps_drive .lt. zero) then
           !G&V model
           if (verbose) write (*,*) "GV model"
           meanB = sum(B)/(Nx*Ny)
           my_rand=ran1(idum_run)
           deltaB = (my_rand/sigma2)**(one/(-sigma1))-one
           itran = 1
           do while ((deltaB .gt. 1e-1*meanB) .and. (itran .le. itrmax))
              my_rand=ran1(idum_run)
              deltaB = (my_rand/sigma2)**(one/(-sigma1))-one
              itran=itran+1
           enddo
           if (itran .ge. itrmax) write (*,*) 'Warning, a delta B could be too large!'
           my_rand_i=ran1(idum_run)
           i = 2+int((nx-2)*my_rand_i)
           my_rand_j=ran1(idum_run)
           j = 2+int((ny-2)*my_rand_j)
           B(i,j) = B(i,j) + deltaB
           lattice_energy(iter)  = sum(B*B)/e0
           energy_released(iter) = zero
           iter = iter + 1
        else if (eps_drive .eq. zero) then
           ! Lu & Hamilton model
           if (verbose) write (*,*) "LH model"
           my_rand=ran1(idum_run)
           deltaB = (sigma2-sigma1)*my_rand + sigma1
           my_rand_i=ran1(idum_run)
           i = 2+int((nx-2)*my_rand_i)
           my_rand_j=ran1(idum_run)
           j = 2+int((ny-2)*my_rand_j)
           B(i,j) = B(i,j) + deltaB
           lattice_energy(iter)  = sum(B*B)/e0
           energy_released(iter) = zero
           iter = iter + 1
        else
           ! Determinsitic forcing
           if (sigma1 .eq. zero) then
              ! accelerate iteration loop
              !niter_to_do = int(ceiling(log(Zc/maxval(abs(Z)))/log(one+eps_drive)))
              niter_to_do = int(log(Zc/maxval(abs(Z)))/log(one+eps_drive))+1
              if (verbose) then
                 write (*,*) '[DEBUG]: ',log(Zc/maxval(abs(Z))),log(one+eps_drive),log(Zc/maxval(abs(Z)))/log(one+eps_drive)
                 write (*,*) '[DEBUG]: ',Zc,maxval(abs(Z)),niter_to_do
              endif
              if (niter_to_do .eq. 0) niter_to_do = 1 ! just a sanity check
              if (niter_to_do .lt. 0) niter_to_do = Niter-iter+1
              if (niter_to_do .eq. 1) then
                 if (last_iter_oneiter .eq. iter-1) then
                    write (*,*) 'Ouch, doing only 1 iterations, pb is still here!'
                 endif
                 last_iter_oneiter = iter
              endif
              ! check if we do not do too many iterations
              niter_to_do = min(niter_to_do,Niter-iter+1)
              if (verbose) then
                 write (*,*) "Doing",niter_to_do,"iterations at once..."
                 write (*,*) "We are at iter=",iter,'and we want to do ',Niter,' iterations'
                 write (*,*) "It will bring to ",iter+niter_to_do-1,"its in total."
                 write (*,*) '[DEBUG]: ',log(Zc/maxval(abs(Z))),log(one+eps_drive)
                 write (*,*) '[DEBUG]: ',Zc,maxval(abs(Z))
              endif
              lattice_energy(iter) = (sum(B*B)/e0)*(one+eps_drive)**2
              if (niter_to_do .gt. 1) then
                 do iter_tmp = iter+1,iter+niter_to_do-1
                    lattice_energy(iter_tmp)  = lattice_energy(iter)*((one+eps_drive)**(2*(iter_tmp-iter)))
                 enddo
              endif
              do i = 2,Nx-1
                 do j = 2,Ny-1
                    B(i,j) = B(i,j)*((one+eps_drive)**niter_to_do)
                 enddo
              enddo
              energy_released(iter:iter+niter_to_do-1) = zero
              iter=iter+niter_to_do
           else
              ! Do the forcing
              do i = 2,Nx-1
                 do j = 2,Ny-1
                    B(i,j) = B(i,j)*(one+eps_drive)
                 enddo
              enddo
              energy_released(iter)=zero
              lattice_energy(iter)=sum(B*B)/e0
              iter=iter+1
           endif
        endif
     endif

  enddo

end subroutine do_avalanche_generic_NCWEIRD


!! subroutine test_tapenade(Niter, &
!!      Nx,Ny,Binit,Zc,D_nc,eps_drive,sigma1,sigma2,lh_soc,idum_obs,idum_init,&
!!      verbose,binsize,threshold,tw, &
!!      lattice_energy,energy_released,energy_released2,B,Z,gradJ)
!!
!!   implicit none
!!   integer, parameter :: sp = selected_real_kind(6,37) ! single precision
!!   integer, parameter :: dp = selected_real_kind(15,307) ! double precision
!!   integer, parameter :: mp =dp
!!   Integer    :: Nx,Ny
!! !f2py integer optional,intent(in) :: Nx=16
!! !f2py integer optional,intent(in) :: Ny=16
!!   Integer    :: Niter
!! !f2py intent(in) Niter
!!   Real(mp) :: Zc,D_nc,eps_drive,sigma1,sigma2
!! !f2py real(mp) optional, intent(in) :: Zc = 1.0
!! !f2py real(mp) optional, intent(in) :: D_nc=0.98
!! !f2py real(mp) optional, intent(in) :: eps_drive=1.e-7
!! !f2py real(mp) optional, intent(in) :: sigma1=0.0
!! !f2py real(mp) optional, intent(in) :: sigma2=0.0
!!   Real(mp) :: lh_soc
!! !f2py real(mp) optional,intent(in) :: lh_soc=0.0
!!   Integer :: idum_obs
!! !f2py integer optional, intent(in) :: idum_obs = -67209495
!!   Integer :: idum_init
!! !f2py integer optional, intent(in) :: idum_init = -827774
!!   Real(mp) :: lattice_energy(Niter),energy_released(Niter),energy_released2(Niter)
!! !f2py intent(out) lattice_energy
!! !f2py intent(out) energy_released
!! !f2py intent(out) energy_released2
!! !f2py depend(Niter) lattice_energy
!! !f2py depend(Niter) energy_released
!! !f2py depend(Niter) energy_released2
!!   Real(mp) :: B(Nx,Ny)
!! !f2py intent(out) B
!! !f2py depend(Nx,Ny) B
!!   Real(mp) :: Z(Nx,Ny)
!! !f2py intent(out) Z
!! !f2py depend(Nx,Ny) Z
!!   Real(mp) :: gradJ(Nx,Ny)
!! !f2py intent(out) gradJ
!! !f2py depend(Nx,Ny) gradJ
!!   Real(mp) :: Binit(Nx,Ny)
!! !f2py real(mp) optional, intent(in), depend(Nx,Ny) :: Binit = 0.0
!!   Logical :: verbose
!! !f2py logical optional, intent(in) :: verbose=0
!!   Integer  :: binsize
!! !f2py intent(in) binsize = 1
!!   Real(mp) :: threshold
!! !f2py real(mp) optional, intent(in) :: threshold = 0.0
!!   Integer  :: tw
!! !f2py integer optional, intent(in) :: tw = 100
!!
!!   !****** Interface
!!   interface
!!      subroutine mkobs(nt,Ereleased,Erelavg,binsize,threshold)
!!        integer, parameter :: sp = selected_real_kind(6,37)
!!        integer, parameter :: dp = selected_real_kind(15,307)
!!        integer, parameter :: mp = dp
!!        Integer :: nt
!!        real(mp), dimension(nt) :: Ereleased
!!        real(mp), dimension(nt) :: Erelavg
!!        integer,  intent(in), optional :: binsize
!!        real(mp), intent(in), optional :: threshold
!!      end subroutine mkobs
!!   end interface
!!   !********* Local variables
!!   Real(mp) :: calc_c,Jcalc_c
!!   Real(mp), allocatable :: obs_Erelavg(:),tmp_e(:)
!!   Integer :: last_idum
!!
!!   Allocate(obs_Erelavg(Niter))
!!   Allocate(tmp_e(Niter))
!!
!!   ! Do soc to generate observations
!!   call do_avalanche_generic_dp(Niter, &
!!      Nx,Ny,Binit,Zc,D_nc,eps_drive,sigma1,sigma2,lh_soc,idum_obs,verbose, &
!!      lattice_energy,energy_released,energy_released2,B,Z,last_idum)
!!   call mkobs(Niter,energy_released,obs_Erelavg,binsize=binsize,threshold=threshold)
!!
!!   ! Calculate first cost
!!   call calc_cost(Niter, &
!!        Nx,Ny,Binit,Zc,D_nc,eps_drive,sigma1,sigma2,lh_soc,idum_init, &
!!        binsize,threshold,tw,obs_Erelavg,calc_c)
!!   write (*,*) "Initial cost",calc_c
!!   Jcalc_c = 1.0
!!
!!   ! Compute the gradient of the cost function
!!   call calc_cost_b(Niter, &
!!        Nx,Ny,Binit,gradJ,Zc,D_nc,eps_drive,sigma1,sigma2,lh_soc,idum_init, &
!!        binsize,threshold,tw,obs_Erelavg,calc_c,Jcalc_c)
!!
!!   Deallocate(obs_Erelavg)
!!   Deallocate(tmp_e)
!!
!! end subroutine test_tapenade


! Use 4Dvar method to find a new initial condition
subroutine find_new_init(Niter, &
     Nx,Ny,Binit,Zc,D_nc,eps_drive,sigma1,sigma2,lh_soc,idum_init,&
     verbose,binsize,threshold,tw,lambd,n_simplex,temp,idum_anneal,niter_step,n_steps, &
     obs_Erelavg,VersionCost,Emoy,Sigma,Emax,Emin,EVsaved,Nev,algo,new_init,final_grad,final_cost)

  use Minimization
  use Prec
  implicit none
  Integer    :: Nx,Ny
!f2py integer optional,intent(in) :: Nx=16
!f2py integer optional,intent(in) :: Ny=16
  Integer    :: Niter
!f2py intent(in) Niter
  Real(mp) :: Zc,D_nc,eps_drive,sigma1,sigma2
!f2py real(mp) optional, intent(in) :: Zc = 1.0
!f2py real(mp) optional, intent(in) :: D_nc=0.98
!f2py real(mp) optional, intent(in) :: eps_drive=1.e-7
!f2py real(mp) optional, intent(in) :: sigma1=0.0
!f2py real(mp) optional, intent(in) :: sigma2=0.0
  Real(mp) :: lh_soc
!f2py real(mp) optional,intent(in) :: lh_soc=0.0
  Integer :: idum_init
!f2py integer optional, intent(in) :: idum_init = -827774
  Real(mp) :: Binit(Nx,Ny)
!f2py real(mp) optional, intent(in), depend(Nx,Ny) :: Binit = 0.0
  Logical :: verbose
!f2py logical optional, intent(in) :: verbose=0
  Integer  :: binsize
!f2py intent(in) binsize = 1
  Real(mp) :: threshold
!f2py real(mp) optional, intent(in) :: threshold = 0.0
  Integer  :: tw
!f2py integer optional, intent(in) :: tw = 100
  Real(mp) :: lambd
!f2py real(mp) optional, intent(in) :: lambd = 1.e-2
  Integer  :: n_simplex
!f2py integer optional, intent(in) :: n_simplex = 1
  Real(mp) :: temp
!f2py real(mp) optional, intent(in) :: temp = 1.
  Integer  :: idum_anneal
!f2py integer optional, intent(in) :: idum_anneal = -827774
  Integer  :: niter_step
!f2py integer optional, intent(in) :: niter_step = 20000
  Integer  :: n_steps
!f2py integer optional, intent(in) :: n_steps = 25
  Real(mp) :: obs_Erelavg(Niter)
!f2py real(mp) optional, intent(in), depend(Niter) :: obs_Erelavg = 0.0
  Integer  :: VersionCost
!f2py integer optional, intent(in) :: VersionCost=0
  Real(mp) :: Emoy
!f2py real(mp) optional, intent(in) :: Emoy = 1.
  Real(mp) :: Sigma
!f2py real(mp) optional, intent(in) :: Sigma = 1.
  Real(mp) :: Emax
!f2py real(mp) optional, intent(in) :: Emax = 1.
  Real(mp) :: Emin
!f2py real(mp) optional, intent(in) :: Emin = 1.
  Integer :: Nev
!f2py integer optional, intent(in) :: Nev = 1
  Character*(*) :: EVsaved
!f2py intent(in) EVsaved
  Integer  :: algo
!f2py integer optional, intent(in) :: algo = 0
  Real(mp) :: new_init(Nx,Ny)
!f2py intent(out) new_init
!f2py depend(Nx,Ny) new_init
  Real(mp) :: final_grad(Nx,Ny)
!f2py intent(out) final_grad
!f2py depend(Nx,Ny) final_grad
  Real(mp) :: final_cost
!f2py intent(out) final_cost

  !********* Local variables
  Real(mp) :: Gtol, fret
  Integer  :: iter,i,j
  Real(mp), allocatable :: Btmp(:,:)

  ! Allocate tables
  Allocate(Btmp(Nx,Ny))

  do i=1,Nx
     do j=1,Ny
        Btmp(i,j) = Binit(i,j)
     end do
  end do

  ! Fill in the Minimization Module
  cVersionCost = VersionCost

  ! Do a minimization
  iter = 0
  fret = 0
  if (verbose) then
     write (*,*) 'Starting minimization process...'
  endif
  if (algo .eq. 0) then
     Gtol = 1e-11
     ! CG
     if (verbose) write (*,*) 'Minimizing with CG'
     write (*,*) "TO BE CODED AGAIN WITH MODERN SYNTAX"
     !Call frprmn(Niter,Nx,Ny,Zc,D_nc,eps_drive,sigma1,sigma2,lh_soc,idum_init, &
     !     binsize,threshold,tw,obs_Erelavg,Emoy,Sigma,Emax,Emin,&
     !     Gtol,iter,verbose,fret,&
     !     Btmp,final_grad)
  else if (algo .eq. 1) then
     Gtol = 1e-14
     if (verbose) write (*,*) 'Minimizing with QN'
     write (*,*) "TO BE CODED AGAIN WITH MODERN SYNTAX"
     ! QN
     !Call QNsolv(Niter,Nx,Ny,Zc,D_nc,eps_drive,sigma1,sigma2,lh_soc,idum_init, &
     !     binsize,threshold,tw,obs_Erelavg,Emoy,Sigma,Emax,Emin,&
     !     Gtol,iter,verbose,fret,&
     !     Btmp)!,final_grad)
  else if (algo .eq. 2) then
     Gtol = 1e-8
     if (verbose) write (*,*) 'Minimizing with simplex'
     ! Simplex
     Call simplex(Niter,Nx,Ny,Zc,D_nc,eps_drive,sigma1,sigma2,lh_soc,idum_init, &
          binsize,threshold,tw,obs_Erelavg,Emoy,Sigma,&
          Gtol,iter,lambd,n_simplex,&
          Btmp)!,final_grad)
  else if (algo .eq. 3) then
     Gtol = 1e-8
     if (verbose) write (*,*) 'Minimizing with simulated annealing'
     ! Simulated annealing
     Call simanneal(Niter,Nx,Ny,Zc,D_nc,eps_drive,sigma1,sigma2,lh_soc,idum_init, &
          binsize,threshold,tw,obs_Erelavg,Emoy,Sigma,Emax,Emin,&
          Gtol,verbose,lambd,n_simplex,temp,idum_anneal,niter_step,n_steps,&
          Btmp,final_cost)
  else if (algo .eq. 4) then
     Gtol = 1e-8
     if (verbose) write (*,*) 'Minimizing with simulated annealing and EigenVectors decomposition'
     ! Simulated annealing
     Call simannealEV(Niter,Nx,Ny,Zc,D_nc,eps_drive,sigma1,sigma2,lh_soc,idum_init, &
          binsize,threshold,tw,obs_Erelavg,Emoy,Sigma,Emax,Emin,EVsaved,Nev,&
          Gtol,verbose,lambd,n_simplex,temp,idum_anneal,niter_step,n_steps,&
          Btmp,final_cost)
  else
     write (*,*) 'Unkown algorithm'
  endif


  new_init = Btmp

  Deallocate(Btmp)

end subroutine find_new_init

subroutine get_cost(Niter, &
     Nx,Ny,Binit,Zc,D_nc,eps_drive,sigma1,sigma2,lh_soc,idum_init,&
     verbose,binsize,threshold,tw, &
     obs_Erelavg,VersionCost,cost)

  use Prec
  implicit none
  Integer    :: Nx,Ny
!f2py integer optional,intent(in) :: Nx=16
!f2py integer optional,intent(in) :: Ny=16
  Integer    :: Niter
!f2py intent(in) Niter
  Real(mp) :: Zc,D_nc,eps_drive,sigma1,sigma2
!f2py real(mp) optional, intent(in) :: Zc = 1.0
!f2py real(mp) optional, intent(in) :: D_nc=0.98
!f2py real(mp) optional, intent(in) :: eps_drive=1.e-7
!f2py real(mp) optional, intent(in) :: sigma1=0.0
!f2py real(mp) optional, intent(in) :: sigma2=0.0
  Real(mp)  :: lh_soc
!f2py real(mp) optional,intent(in) :: lh_soc=0.0
  Integer :: idum_init
!f2py integer optional, intent(in) :: idum_init = -827774
  Real(mp) :: Binit(Nx,Ny)
!f2py real(mp) optional, intent(in), depend(Nx,Ny) :: Binit = 0.0
  Logical :: verbose
!f2py logical optional, intent(in) :: verbose=0
  Integer  :: binsize
!f2py intent(in) binsize = 1
  Real(mp) :: threshold
!f2py real(mp) optional, intent(in) :: threshold = 0.0
  Integer  :: tw
!f2py integer optional, intent(in) :: tw = 100
  Real(mp) :: obs_Erelavg(Niter)
!f2py real(mp) optional, intent(in), depend(Niter) :: obs_Erelavg = 0.0
  Integer  :: VersionCost
!f2py integer optional, intent(in) :: VersionCost = 0
  Real(mp) :: cost
!f2py real(mp), intent(out) :: cost

  Call calc_cost(Niter, &
       Nx,Ny,Binit,Zc,D_nc,eps_drive,sigma1,sigma2,lh_soc,idum_init, &
       binsize,threshold,tw,obs_Erelavg,VersionCost,cost)
  if (verbose) then
     write (*,*) 'Calculated cost:',cost
  endif

end subroutine get_cost

subroutine get_only_cost(Niter, E_relavg, obs_Erelavg, &
     threshold,tw,cost)

  use Prec
  implicit none
  Integer    :: Niter
!f2py intent(in) Niter
  Real(mp) :: threshold
!f2py real(mp) optional, intent(in) :: threshold = 0.0
  Integer  :: tw
!f2py integer optional, intent(in) :: tw = 100
  Real(mp) :: E_relavg(Niter)
!f2py real(mp) optional, intent(in), depend(Niter) :: E_relavg = 0.0
  Real(mp) :: obs_Erelavg(Niter)
!f2py real(mp) optional, intent(in), depend(Niter) :: obs_Erelavg = 0.0
  Integer  :: VersionCost
!f2py integer optional, intent(in) :: VersionCost = 0
  Real(mp) :: cost
!f2py real(mp), intent(out) :: cost

  Call exotic_cost(Niter, E_relavg,obs_Erelavg, &
       tw,cost,threshold,VersionCost)

end subroutine get_only_cost
