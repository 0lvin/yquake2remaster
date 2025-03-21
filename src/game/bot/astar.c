/*
 * Copyright (C) 1997-2001 Id Software, Inc.
 * Copyright (C) 2001 Steve Yeager
 * Copyright (C) 2001-2004 Pat AfterMoon
 * Copyright (c) ZeniMax Media Inc.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA
 * 02111-1307, USA.
 *
 */

#include "../header/local.h"
#include "ai_local.h"

//==========================================
//
//
//==========================================

static int alist[MAX_NODES];	//list contains all studied nodes, Open and Closed together
static int alist_numNodes;

enum {
	NOLIST,
	OPENLIST,
	CLOSEDLIST
};

typedef struct
{
	int	parent;
	int		G;
	int		H;

	int	list;

} astarnode_t;

static astarnode_t	astarnodes[MAX_NODES];

struct astarpath_s *Apath;
//==========================================
//
//
//==========================================
static int originNode;
static int goalNode;
static int currentNode;

static int ValidLinksMask;
#define DEFAULT_MOVETYPES_MASK ( \
	LINK_MOVE | \
	LINK_STAIRS | \
	LINK_FALL | \
	LINK_WATER | \
	LINK_WATERJUMP | \
	LINK_JUMPPAD | \
	LINK_PLATFORM | \
	LINK_TELEPORT)
//==========================================
//
//
//
//==========================================

static qboolean
AStar_nodeIsInClosed(int node)
{
	if (node >= MAX_NODES || node < 0)
	{
		return false;
	}

	if (astarnodes[node].list == CLOSEDLIST)
	{
		return true;
	}

	return false;
}

static qboolean
AStar_nodeIsInOpen(int node)
{
	if (node >= MAX_NODES || node < 0)
	{
		return false;
	}

	if(astarnodes[node].list == OPENLIST)
	{
		return true;
	}

	return false;
}

static void
AStar_InitLists(void)
{
	size_t i;

	for (i = 0; i < MAX_NODES; i++)
	{
		astarnodes[i].G = 0;
		astarnodes[i].H = 0;
		astarnodes[i].parent = 0;
		astarnodes[i].list = NOLIST;
	}

	if(Apath)
	{
		Apath->numNodes = 0;
	}

	alist_numNodes = 0;
	memset(alist, -1, sizeof(alist));//jabot092
}

static int
AStar_PLinkDistance(int n1, int n2)
{
	size_t i;

	for (i = 0; i < pLinks[n1].numLinks; i++)
	{
		if (pLinks[n1].nodes[i] == n2)
		{
			return (int)pLinks[n1].dist[i];
		}
	}

	return -1;
}

static int
Astar_HDist_ManhatanGuess(int node)
{
	vec3_t DistVec;
	size_t i;

	//teleporters are exceptional
	if (nodes[node].flags & NODEFLAGS_TELEPORTER_IN)
	{
		node++; //it's tele out is stored in the next node in the array
	}

	for (i = 0 ; i < 3 ; i++)
	{
		DistVec[i] = fabs(nodes[goalNode].origin[i] - nodes[node].origin[i]);
	}

	return (DistVec[0] + DistVec[1] + DistVec[2]);
}

static void
AStar_PutInClosed(int node)
{
	if (!astarnodes[node].list)
	{
		alist[alist_numNodes] = node;
		alist_numNodes++;
	}

	astarnodes[node].list = CLOSEDLIST;
}

static void
AStar_PutAdjacentsInOpen(int node)
{
	size_t i;

	for (i = 0; i < pLinks[node].numLinks; i++)
	{
		int addnode;

		// ignore invalid links
		if (!(ValidLinksMask & pLinks[node].moveType[i]))
		{
			continue;
		}

		addnode = pLinks[node].nodes[i];

		// ignore self
		if (addnode == node)
		{
			continue;
		}

		// ignore if it's already in closed list
		if (AStar_nodeIsInClosed(addnode))
		{
			continue;
		}

		// if it's already inside open list
		if (AStar_nodeIsInOpen(addnode))
		{
			int plinkDist;

			plinkDist = AStar_PLinkDistance(node, addnode);
			if (plinkDist == -1 && bot_debugmonster->value)
			{
				Com_Printf("WARNING: AStar_PutAdjacentsInOpen - Couldn't find distance between nodes\n");
			}
			// compare G distances and choose best parent
			else if (astarnodes[addnode].G > (astarnodes[node].G + plinkDist))
			{
				astarnodes[addnode].parent = node;
				astarnodes[addnode].G = astarnodes[node].G + plinkDist;
			}
		}
		else
		{
			// just put it in
			int plinkDist;

			plinkDist = AStar_PLinkDistance( node, addnode );
			if (plinkDist == -1)
			{
				plinkDist = AStar_PLinkDistance(addnode, node);
				if (plinkDist == -1)
				{
					plinkDist = 999;//jalFIXME
				}

				if (bot_debugmonster->value)
				{
					Com_Printf("WARNING: AStar_PutAdjacentsInOpen - Couldn't find distance between nodes\n");
				}
			}

			// put in global list
			if (!astarnodes[addnode].list)
			{
				alist[alist_numNodes] = addnode;
				alist_numNodes++;
			}

			astarnodes[addnode].parent = node;
			astarnodes[addnode].G = astarnodes[node].G + plinkDist;
			astarnodes[addnode].H = Astar_HDist_ManhatanGuess( addnode );
			astarnodes[addnode].list = OPENLIST;
		}
	}
}

static int
AStar_FindInOpen_BestF(void)
{
	size_t i;
	int	bestF = -1;
	int best = -1;

	for (i = 0; i < alist_numNodes; i++)
	{
		int node = alist[i];

		if( astarnodes[node].list != OPENLIST )
		{
			continue;
		}

		if (bestF == -1 || bestF > (astarnodes[node].G + astarnodes[node].H))
		{
			bestF = astarnodes[node].G + astarnodes[node].H;
			best = node;
		}
	}

	if (bot_debugmonster->value)
	{
		Com_Printf("BEST:%i\n", best);
	}

	return best;
}

static void
AStar_ListsToPath(void)
{
	int count = 0;
	int cur = goalNode;
	int *pnode;

	Apath->numNodes = 0;
	pnode = Apath->nodes;
	while (cur != originNode)
	{
		*pnode = cur;
		pnode++;
		cur = astarnodes[cur].parent;
		count++;
	}

	Apath->numNodes = count - 1;
}

static qboolean
AStar_FillLists(void)
{
	// put current node inside closed list
	AStar_PutInClosed(currentNode);

	// put adjacent nodes inside open list
	AStar_PutAdjacentsInOpen(currentNode);

	// find best adjacent and make it our current
	currentNode = AStar_FindInOpen_BestF();

	return (currentNode != -1);	//if -1 path is bloqued
}

static qboolean
AStar_ResolvePath(int origin, int goal, int movetypes)
{
	if (origin < 0 || goal < 0)
	{
		return false;
	}

	ValidLinksMask = movetypes;
	if (!ValidLinksMask)
	{
		ValidLinksMask = DEFAULT_MOVETYPES_MASK;
	}

	AStar_InitLists();

	currentNode = originNode = origin;
	goalNode = goal;

	while (!AStar_nodeIsInOpen(goalNode))
	{
		if(!AStar_FillLists())
		{
			return false;	//failed
		}
	}

	AStar_ListsToPath();

	return true;
}

qboolean
AStar_GetPath(int origin, int goal, int movetypes, struct astarpath_s *path)
{
	Apath = path;

	if( !AStar_ResolvePath ( origin, goal, movetypes ) )
	{
		return false;
	}

	path->originNode = origin;
	path->goalNode = goal;
	return true;
}
