# SJE with stochastic gradient

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111, USA
#
#
# Zeynep Akata
# e-mail: akata@mpi-inf.mpg.de
# 
# Max Planck Institute for Informatics
# Campus E1 4
# 66123 Saarbrucken
# GERMANY


L=lib
BIN=../bin
CXX=g++
OPT=
#OPTS=-g -O2
OPTS=-g
CXXFLAGS= ${OPTS} ${OPT} -Wall -I$L
LIBS = -lz -lm

PROGRAMS = ${BIN}/prep_data ${BIN}/msje_lowrank

OBJS = vectors.o gzstream.o timer.o
INCS = $L/vectors.h $L/gzstream.h $L/timer.h $L/wrapper.h $L/assert.h

all: ${PROGRAMS}

clean:
	-rm ${PROGRAMS} 
	-rm *.o 

vectors.o: $L/vectors.cpp ${INCS}
	${CXX} ${CXXFLAGS} -c -o $@ $L/vectors.cpp

gzstream.o: $L/gzstream.cpp ${INCS}
	${CXX} ${CXXFLAGS} -c -o $@ $L/gzstream.cpp

timer.o: $L/timer.cpp ${INCS}
	${CXX} ${CXXFLAGS} -c -o $@ $L/timer.cpp

data_mult.o: data_mult.cpp ${INCS}
	${CXX} ${CXXFLAGS} -c -o $@ data_mult.cpp

prep_data.o: prep_data.cpp ${INCS}
	${CXX} ${CXXFLAGS} -c -o $@ prep_data.cpp


msje_lowrank.o: msje_lowrank.cpp data_mult.h loss.h ${INCS}
	${CXX} ${CXXFLAGS} -c -o $@ msje_lowrank.cpp


${BIN}/prep_data: prep_data.o ${OBJS}
	${CXX} ${CXXFLAGS} -o $@ prep_data.o ${OBJS} ${LIBS}


${BIN}/msje_lowrank: msje_lowrank.o data_mult.o ${OBJS}
	${CXX} ${CXXFLAGS} -o $@ msje_lowrank.o data_mult.o ${OBJS} ${LIBS}
