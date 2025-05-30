/*
 * Copyright (C) 1997-2001 Id Software, Inc.
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
 * =======================================================================
 *
 * This file implements interpolation between two frames. This is used
 * to smooth down network play
 *
 * =======================================================================
 */

#include "header/client.h"

void
CL_CheckPredictionError(void)
{
	int frame;
	int delta[3];
	int i;
	int len;

	if (!cl_predict->value ||
		(cl.frame.playerstate.pmove.pm_flags & PMF_NO_PREDICTION))
	{
		return;
	}

	/* calculate the last usercmd_t we sent that the server has processed */
	frame = cls.netchan.incoming_acknowledged;
	frame &= (CMD_BACKUP - 1);

	/* compare what the server returned with what we had predicted it to be */
	VectorSubtract(cl.frame.origin,
			cl.predicted_origins[frame], delta);

	/* save the prediction error for interpolation */
	len = abs(delta[0]) + abs(delta[1]) + abs(delta[2]);

	/* 80 world units */
	if (len > 640)
	{
		/* a teleport or something */
		VectorClear(cl.prediction_error);
	}
	else
	{
		if (cl_showmiss->value && (delta[0] || delta[1] || delta[2]))
		{
			Com_Printf("prediction miss on %i: %i\n", cl.frame.serverframe,
					delta[0] + delta[1] + delta[2]);
		}

		VectorCopy(cl.frame.origin,
				cl.predicted_origins[frame]);

		/* save for error itnerpolation */
		for (i = 0; i < 3; i++)
		{
			cl.prediction_error[i] = delta[i] * 0.125f;
		}
	}
}

void
CL_ClipMoveToEntities(vec3_t start, vec3_t mins, vec3_t maxs,
		vec3_t end, trace_t *tr)
{
	int i;

	for (i = 0; i < cl.frame.num_entities; i++)
	{
		int x, zd, zu;
		trace_t trace;
		int headnode;
		float *angles;
		int num;
		cmodel_t *cmodel;
		vec3_t bmins, bmaxs;
		entity_xstate_t *ent;

		num = (cl.frame.parse_entities + i) & (MAX_PARSE_ENTITIES - 1);
		ent = &cl_parse_entities[num];

		if (!ent->solid)
		{
			continue;
		}

		if (ent->number == cl.playernum + 1)
		{
			continue;
		}

		if (ent->solid == 31)
		{
			/* special value for bmodel */
			cmodel = cl.model_clip[ent->modelindex];

			if (!cmodel)
			{
				continue;
			}

			headnode = cmodel->headnode;
			angles = ent->angles;
		}
		else
		{
			/* encoded bbox */
			x = 8 * (ent->solid & 31);
			zd = 8 * ((ent->solid >> 5) & 31);
			zu = 8 * ((ent->solid >> 10) & 63) - 32;

			bmins[0] = bmins[1] = -(float)x;
			bmaxs[0] = bmaxs[1] = (float)x;
			bmins[2] = -(float)zd;
			bmaxs[2] = (float)zu;

			headnode = CM_HeadnodeForBox(bmins, bmaxs);
			angles = vec3_origin; /* boxes don't rotate */
		}

		if (tr->allsolid)
		{
			return;
		}

		trace = CM_TransformedBoxTrace(start, end,
				mins, maxs, headnode, MASK_PLAYERSOLID,
				ent->origin, angles);

		if (trace.allsolid || trace.startsolid ||
			(trace.fraction < tr->fraction))
		{
			trace.ent = (struct edict_s *)ent;

			if (tr->startsolid)
			{
				*tr = trace;
				tr->startsolid = true;
			}
			else
			{
				*tr = trace;
			}
		}
	}
}

trace_t
CL_PMTrace(vec3_t start, vec3_t mins, vec3_t maxs, vec3_t end)
{
	trace_t t;

	/* check against world */
	t = CM_BoxTrace(start, end, mins, maxs, 0, MASK_PLAYERSOLID);

	if (t.fraction < 1.0)
	{
		t.ent = (struct edict_s *)1;
	}

	/* check all other solid models */
	CL_ClipMoveToEntities(start, mins, maxs, end, &t);

	return t;
}

int
CL_PMpointcontents(vec3_t point)
{
	int i;
	int contents;

	contents = CM_PointContents(point, 0);

	for (i = 0; i < cl.frame.num_entities; i++)
	{
		entity_xstate_t *ent;
		int num;
		cmodel_t *cmodel;

		num = (cl.frame.parse_entities + i) & (MAX_PARSE_ENTITIES - 1);
		ent = &cl_parse_entities[num];

		if (ent->solid != 31) /* special value for bmodel */
		{
			continue;
		}

		cmodel = cl.model_clip[ent->modelindex];

		if (!cmodel)
		{
			continue;
		}

		contents |= CM_TransformedPointContents(point, cmodel->headnode,
				ent->origin, ent->angles);
	}

	return contents;
}

/*
 * Sets cl.predicted_origin and cl.predicted_angles
 */
void
CL_PredictMovement(void)
{
	int ack, current, origin[3];
	usercmd_t *cmd;
	pmove_t pm;
	int i;
	int step;
	vec3_t tmp;

	if (cls.state != ca_active)
	{
		return;
	}

	if (cl_paused->value)
	{
		return;
	}

	if (!cl_predict->value ||
		(cl.frame.playerstate.pmove.pm_flags & PMF_NO_PREDICTION))
	{
		/* just set angles */
		for (i = 0; i < 3; i++)
		{
			cl.predicted_angles[i] = cl.viewangles[i] + SHORT2ANGLE(
					cl.frame.playerstate.pmove.delta_angles[i]);
		}

		return;
	}

	ack = cls.netchan.incoming_acknowledged;
	current = cls.netchan.outgoing_sequence;

	/* if we are too far out of date, just freeze */
	if (current - ack >= CMD_BACKUP)
	{
		if (cl_showmiss->value)
		{
			Com_Printf("exceeded CMD_BACKUP\n");
		}

		return;
	}

	/* copy current state to pmove */
	memset (&pm, 0, sizeof(pm));
	pm.trace = CL_PMTrace;
	pm.pointcontents = CL_PMpointcontents;
	pm_airaccelerate = atof(cl.configstrings[CS_AIRACCEL]);
	pm.s = cl.frame.playerstate.pmove;

	VectorCopy(cl.frame.origin, origin);

	/* run frames */
	while (++ack <= current)
	{
		int frame;

		frame = ack & (CMD_BACKUP - 1);
		cmd = &cl.cmds[frame];

		// Ignore null entries
		if (!cmd->msec)
		{
			continue;
		}

		pm.cmd = *cmd;
		PmoveEx(&pm, origin);

		/* save for debug checking */
		VectorCopy(origin, cl.predicted_origins[frame]);
	}

	// step is used for movement prediction on stairs
	// (so moving up/down stairs is smooth)
	step = origin[2] - (int)(cl.predicted_origin[2] * 8);
	VectorCopy(pm.s.velocity, tmp);

	if (((step > 62 && step < 66) || (step > 94 && step < 98) || (step > 126 && step < 130))
		&& !VectorCompare(tmp, vec3_origin)
		&& (pm.s.pm_flags & PMF_ON_GROUND))
	{
		cl.predicted_step = step * 0.125f;
		cl.predicted_step_time = cls.realtime - (int)(cls.nframetime * 500);
	}

	/* copy results out for rendering */
	cl.predicted_origin[0] = origin[0] * 0.125f;
	cl.predicted_origin[1] = origin[1] * 0.125f;
	cl.predicted_origin[2] = origin[2] * 0.125f;

	VectorCopy(pm.viewangles, cl.predicted_angles);
}

