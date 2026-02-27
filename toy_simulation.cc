/* -*-  Mode: C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2017 Universität zu Lübeck [GEYER]
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
 *
 * Author: Regine Geyer <geyer@itm.uni-luebeck.de>
 */

#include "ns3/Bloodcircuit.h"
#include "ns3/core-module.h"

using namespace ns3;
using namespace std;

int
main (int argc, char *argv[])
{
  int numOfNanobots = 64;
  int simulationDuration = 1800;
  int injectionVessel = 15;          

  CommandLine cmd;
  cmd.AddValue ("simulationDuration", "Total duration of the simulation in seconds", simulationDuration);
  cmd.AddValue ("numOfNanobots", "Total number of nanodevices in the bloodstream", numOfNanobots);
  cmd.AddValue ("injectionVessel", "The ID of the vessel where nanobots are injected", injectionVessel);
  cmd.Parse (argc, argv);

  return Bloodcircuit::BeginnSimulation (simulationDuration, numOfNanobots, injectionVessel);
}