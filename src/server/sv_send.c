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
 * Message sending and multiplexing.
 *
 * =======================================================================
 */

#include "header/server.h"

char sv_outputbuf[SV_OUTPUTBUF_LENGTH];

void
SV_FlushRedirect(int sv_redirected, char *outputbuf)
{
	if (sv_redirected == RD_PACKET)
	{
		Netchan_OutOfBandPrint(NS_SERVER, net_from, "print\n%s", outputbuf);
	}
	else if (sv_redirected == RD_CLIENT)
	{
		MSG_WriteByte(&sv_client->netchan.message, svc_print);
		MSG_WriteByte(&sv_client->netchan.message, PRINT_HIGH);
		MSG_WriteString(&sv_client->netchan.message, outputbuf);
	}
}

/*
 * Sends text across to be displayed if the level passes
 */
void
SV_ClientPrintf(client_t *cl, int level, const char *fmt, ...)
{
	va_list argptr;
	char string[1024];

	va_start(argptr, fmt);
	vsnprintf(string, sizeof(string), fmt, argptr);
	va_end(argptr);

	MSG_WriteByte(&cl->netchan.message, svc_print);
	MSG_WriteByte(&cl->netchan.message, level);
	MSG_WriteString(&cl->netchan.message, string);
}

/*
 * Sends text to all active clients
 */
void
SV_BroadcastPrintf(int level, const char *fmt, ...)
{
	va_list argptr;
	char string[2048];
	client_t *cl;
	int i;

	va_start(argptr, fmt);
	vsnprintf(string, sizeof(string), fmt, argptr);
	va_end(argptr);

	/* echo to console */
	if (dedicated->value)
	{
		char copy[1024];
		int i;

		/* mask off high bits */
		for (i = 0; i < 1023 && string[i]; i++)
		{
			copy[i] = string[i] & 127;
		}

		copy[i] = 0;
		Com_Printf("%s", copy);
	}

	for (i = 0, cl = svs.clients; i < maxclients->value; i++, cl++)
	{
		if (cl->state != cs_spawned)
		{
			continue;
		}

		MSG_WriteByte(&cl->netchan.message, svc_print);
		MSG_WriteByte(&cl->netchan.message, level);
		MSG_WriteString(&cl->netchan.message, string);
	}
}

/*
 * Sends text to all active clients
 */
void
SV_BroadcastCommand(const char *fmt, ...)
{
	va_list argptr;
	char string[1024];

	if (!sv.state)
	{
		return;
	}

	va_start(argptr, fmt);
	vsnprintf(string, sizeof(string), fmt, argptr);
	va_end(argptr);

	MSG_WriteByte(&sv.multicast, svc_stufftext);
	MSG_WriteString(&sv.multicast, string);
	SV_Multicast(NULL, MULTICAST_ALL_R);
}

/*
 * Sends the contents of sv.multicast to a subset of the clients,
 * then clears sv.multicast.
 *
 * MULTICAST_ALL	same as broadcast (origin can be NULL)
 * MULTICAST_PVS	send to clients potentially visible from org
 * MULTICAST_PHS	send to clients potentially hearable from org
 */
static qboolean
SV_WereConnected(const vec3_t origin, const byte *mask, int area1)
{
	vec3_t origin2;
	int leafnum;
	int cluster;

	VectorCopy(origin, origin2);

	leafnum = CM_PointLeafnum(origin2);
	cluster = CM_LeafCluster(leafnum);

	// cluster can be -1 if we're in the void (or sometimes just at a wall)
	// and using a negative index into mask[] would be invalid
	if (cluster >= 0 && (mask[cluster >> 3] & (1 << (cluster & 7))))
	{
		if (CM_AreasConnected(area1, CM_LeafArea(leafnum)))
		{
			return true;
		}
	}

	// if the client is currently in water, do a second check
	if (CM_PointContents(origin2, 0) & MASK_WATER)
	{
		// if the client is half-submerged in opaque water so its origin
		// is below the water, but the head/camera is still above the water
		// and thus should be able to see/hear explosions or similar
		// that are above the water.
		// so try again at a slightly higher position
		// FIXME: OTOH, we have a similar problem if we're over water and shoot under water (near water level) => can't see explosion

		origin2[2] += 32.0f;

		leafnum = CM_PointLeafnum(origin2);
		cluster = CM_LeafCluster(leafnum);

		if (cluster >= 0 && (mask[cluster >> 3] & (1 << (cluster & 7))))
		{
			if (CM_AreasConnected(area1, CM_LeafArea(leafnum)))
			{
				return true;
			}
		}
	}

	return false;
}

void
SV_Multicast(vec3_t origin, multicast_t to)
{
	client_t *client;
	byte *mask;
	int leafnum, cluster;
	int j;
	qboolean reliable;
	int area1;

	reliable = false;

	if ((to != MULTICAST_ALL_R) && (to != MULTICAST_ALL))
	{
		leafnum = CM_PointLeafnum(origin);
		area1 = CM_LeafArea(leafnum);
	}
	else
	{
		leafnum = 0;
		area1 = 0;
	}

	/* if doing a serverrecord, store everything */
	if (svs.demofile)
	{
		SZ_Write(&svs.demo_multicast, sv.multicast.data, sv.multicast.cursize);
	}

	switch (to)
	{
		case MULTICAST_ALL_R:
			reliable = true; /* intentional fallthrough */
		case MULTICAST_ALL:
			mask = NULL;
			break;

		case MULTICAST_PHS_R:
			reliable = true; /* intentional fallthrough */
		case MULTICAST_PHS:
			leafnum = CM_PointLeafnum(origin);
			cluster = CM_LeafCluster(leafnum);
			mask = CM_ClusterPHS(cluster);
			break;

		case MULTICAST_PVS_R:
			reliable = true; /* intentional fallthrough */
		case MULTICAST_PVS:
			leafnum = CM_PointLeafnum(origin);
			cluster = CM_LeafCluster(leafnum);
			mask = CM_ClusterPVS(cluster);
			break;

		default:
			mask = NULL;
			Com_Error(ERR_FATAL, "%s: bad to:%i", __func__, to);
	}

	/* send the data to all relevent clients */
	for (j = 0, client = svs.clients; j < maxclients->value; j++, client++)
	{
		if ((client->state == cs_free) || (client->state == cs_zombie))
		{
			continue;
		}

		if ((client->state != cs_spawned) && !reliable)
		{
			continue;
		}

		if (mask)
		{
			if (!SV_WereConnected(client->edict->s.origin, mask, area1))
			{
				continue;
			}
		}

		SZ_Write(reliable ? &client->netchan.message : &client->datagram,
				sv.multicast.data, sv.multicast.cursize);
	}

	SZ_Clear(&sv.multicast);
}

/*
 * Each entity can have eight independant sound sources, like voice,
 * weapon, feet, etc.
 *
 * If cahnnel & 8, the sound will be sent to everyone, not just
 * things in the PHS.
 *
 * Channel 0 is an auto-allocate channel, the others override anything
 * already running on that entity/channel pair.
 *
 * An attenuation of 0 will play full volume everywhere in the level.
 * Larger attenuations will drop off.  (max 4 attenuation)
 *
 * Timeofs can range from 0.0 to 0.1 to cause sounds to be started
 * later in the frame than they normally would.
 *
 * If origin is NULL, the origin is determined from the entity origin
 * or the midpoint of the entity box for bmodels.
 */
void
SV_StartSound(vec3_t origin, edict_t *entity, int channel, int soundindex,
		float volume, float attenuation, float timeofs)
{
	int sendchan, flags, i, ent, protocol;
	qboolean use_phs;
	vec3_t origin_v;

	protocol = sv_client ? sv_client->protocol : PROTOCOL_VERSION;

	if ((volume < 0) || (volume > 1.0))
	{
		Com_Error(ERR_FATAL, "%s: volume = %f", __func__, volume);
	}

	if ((attenuation < 0) || (attenuation > 4))
	{
		Com_Error(ERR_FATAL, "%s: attenuation = %f", __func__, attenuation);
	}

	if ((timeofs < 0) || (timeofs > 0.255))
	{
		Com_Error(ERR_FATAL, "%s: timeofs = %f", __func__, timeofs);
	}

	ent = NUM_FOR_EDICT(entity);

	if (channel & 8) /* no PHS flag */
	{
		use_phs = false;
		channel &= 7;
	}
	else
	{
		use_phs = true;
	}

	sendchan = (ent << 3) | (channel & 7);

	flags = 0;

	if (volume != DEFAULT_SOUND_PACKET_VOLUME)
	{
		flags |= SND_VOLUME;
	}

	if (attenuation != DEFAULT_SOUND_PACKET_ATTENUATION)
	{
		flags |= SND_ATTENUATION;
	}

	/* the client doesn't know that bmodels have
	   weird origins the origin can also be
	   explicitly set */
	if ((entity->svflags & SVF_NOCLIENT) ||
		(entity->solid == SOLID_BSP) ||
		origin)
	{
		flags |= SND_POS;
	}

	/* always send the entity number for channel overrides */
	flags |= SND_ENT;

	if (timeofs)
	{
		flags |= SND_OFFSET;
	}

	/* use the entity origin unless it is a bmodel or explicitly specified */
	if (!origin)
	{
		origin = origin_v;

		if (entity->solid == SOLID_BSP)
		{
			for (i = 0; i < 3; i++)
			{
				origin_v[i] = entity->s.origin[i] + 0.5f *
							  (entity->mins[i] + entity->maxs[i]);
			}
		}
		else
		{
			VectorCopy(entity->s.origin, origin_v);
		}
	}

	MSG_WriteByte(&sv.multicast, svc_sound);
	MSG_WriteByte(&sv.multicast, flags);
	if (IS_QII97_PROTOCOL(protocol))
	{
		MSG_WriteByte(&sv.multicast, soundindex);
	}
	else
	{
		MSG_WriteShort(&sv.multicast, soundindex);
	}

	if (flags & SND_VOLUME)
	{
		MSG_WriteByte(&sv.multicast, volume * 255);
	}

	if (flags & SND_ATTENUATION)
	{
		MSG_WriteByte(&sv.multicast, attenuation * 64);
	}

	if (flags & SND_OFFSET)
	{
		MSG_WriteByte(&sv.multicast, timeofs * 1000);
	}

	if (flags & SND_ENT)
	{
		MSG_WriteShort(&sv.multicast, sendchan);
	}

	if (flags & SND_POS)
	{
		MSG_WritePos(&sv.multicast, origin, protocol);
	}

	/* if the sound doesn't attenuate,send it to everyone
	   (global radio chatter, voiceovers, etc) */
	if (attenuation == ATTN_NONE)
	{
		use_phs = false;
	}

	if (channel & CHAN_RELIABLE)
	{
		if (use_phs)
		{
			SV_Multicast(origin, MULTICAST_PHS_R);
		}
		else
		{
			SV_Multicast(origin, MULTICAST_ALL_R);
		}
	}
	else
	{
		if (use_phs)
		{
			SV_Multicast(origin, MULTICAST_PHS);
		}
		else
		{
			SV_Multicast(origin, MULTICAST_ALL);
		}
	}
}

static int msgbuff_size = 0;
static byte *msgbuff_cache = NULL;

static byte *
SV_SendReallocBuffers(int *num)
{
	void *ptr;

	if (*num < msgbuff_size)
	{
		*num = msgbuff_size;
		return msgbuff_cache;
	}

	msgbuff_size = *num * 2;
	ptr = realloc(msgbuff_cache, msgbuff_size);
	if (!ptr)
	{
		Com_Error(ERR_FATAL, "%s: can't allocate memory", __func__);
		return NULL;
	}

	msgbuff_cache = ptr;

	Com_DPrintf("%s: Realloc send buffer: %d\n", __func__, msgbuff_size);

	*num = msgbuff_size;
	return msgbuff_cache;
}

void
SV_SendInitBuffers(void)
{
	int size;

	msgbuff_size = 0;
	msgbuff_cache = NULL;
	size = MAX_MSGLEN;

	SV_SendReallocBuffers(&size);
}

void
SV_SendFreeBuffers(void)
{
	if (msgbuff_cache)
	{
		free(msgbuff_cache);
		msgbuff_cache = NULL;
	}
	msgbuff_size = 0;
}

static qboolean
SV_SendClientDatagram(client_t *client)
{
	int msg_buf_size;
	byte *msg_buf;
	sizebuf_t msg;

	msg_buf_size = MAX_MSGLEN;
	msg_buf = SV_SendReallocBuffers(&msg_buf_size);

	SV_BuildClientFrame(client);

	SZ_Init(&msg, msg_buf, msg_buf_size);
	msg.allowoverflow = true;

	/* send over all the relevant entity_state_t
	   and the player_state_t */
	SV_WriteFrameToClient(client, &msg);

	/* copy the accumulated multicast datagram
	   for this client out to the message
	   it is necessary for this to be after the WriteEntities
	   so that entity references will be current */
	if (client->datagram.overflowed)
	{
		Com_Printf("WARNING: datagram overflowed for %s\n", client->name);
	}
	else
	{
		SZ_Write(&msg, client->datagram.data, client->datagram.cursize);
	}

	SZ_Clear(&client->datagram);

	if (msg.overflowed)
	{
		/* must have room left for the packet header */
		Com_Printf("WARNING: msg overflowed for %s\n", client->name);
		SZ_Clear(&msg);
	}

	/* send the datagram */
	Netchan_Transmit(&client->netchan, msg.cursize, msg.data);

	/* record the size for rate estimation */
	client->message_size[sv.framenum % RATE_MESSAGES] = msg.cursize;

	return true;
}

static void
SV_DemoCompleted(void)
{
	if (sv.demofile)
	{
		FS_FCloseFile(sv.demofile);
		sv.demofile = 0;
	}

	SV_Nextserver();
}

/*
 * Returns true if the client is over its current
 * bandwidth estimation and should not be sent another packet
 */
static qboolean
SV_RateDrop(client_t *c)
{
	int total;
	int i;

	/* never drop over the loopback */
	if (c->netchan.remote_address.type == NA_LOOPBACK)
	{
		return false;
	}

	total = 0;

	for (i = 0; i < RATE_MESSAGES; i++)
	{
		total += c->message_size[i];
	}

	if (total > c->rate)
	{
		c->surpressCount++;
		c->message_size[sv.framenum % RATE_MESSAGES] = 0;
		return true;
	}

	return false;
}

static int
SV_NextDemoChunk(byte **msgbuf)
{
	size_t r;
	int n, msg_buf_size;;

	if (sv_paused->value)
	{
		return 0;
	}

	r = FS_FRead(&n, 4, 1, sv.demofile);

	if (r != 4)
	{
		return -1;
	}

	n = LittleLong(n);

	if (n == -1)
	{
		return -1;
	}

	msg_buf_size = Q_max(n, MAX_MSGLEN);
	*msgbuf = SV_SendReallocBuffers(&msg_buf_size);

	if (n > MAX_MSGLEN)
	{
		Com_Printf("%s: msglen %d > MAX_MSGLEN\n", __func__, n);
	}

	r = FS_FRead(*msgbuf, n, 1, sv.demofile);

	return (r == n) ? n : -1;
}

/* if the reliable message
   overflowed, drop the
   client */
static void
SV_SendDisconnect(client_t *c)
{
	SZ_Clear(&c->netchan.message);
	SZ_Clear(&c->datagram);

	SV_BroadcastPrintf(PRINT_HIGH, "%s overflowed\n", c->name);
	SV_DropClient(c);

	Netchan_Transmit(&c->netchan, 0, NULL);
}

void
SV_SendClientMessages(void)
{
	int i;
	client_t *c;
	int msglen;
	byte *msgbuf = NULL;

	msglen = 0;

	/* read the next demo message if needed */
	if (sv.demofile && (sv.state == ss_demo))
	{
		msglen = SV_NextDemoChunk(&msgbuf);

		if (msglen < 0)
		{
			SV_DemoCompleted();
			return;
		}
	}
	else
	{
		msglen = 0;
	}

	/* send a message to each spawned client */
	for (i = 0, c = svs.clients; i < maxclients->value; i++, c++)
	{
		if (c->state == cs_free)
		{
			continue;
		}

		if (c->netchan.message.overflowed)
		{
			SV_SendDisconnect(c);
			continue;
		}

		if ((sv.state == ss_cinematic) ||
			(sv.state == ss_demo) ||
			(sv.state == ss_pic))
		{
			Netchan_Transmit(&c->netchan, msglen, msgbuf);
		}
		else if (c->state == cs_spawned)
		{
			/* don't overrun bandwidth */
			if (SV_RateDrop(c))
			{
				continue;
			}

			SV_SendClientDatagram(c);
		}

		/* messages to non-spawned clients are sent by SendPrepClientMessages */
	}
}

void
SV_SendPrepClientMessages(void)
{
	client_t *c;
	int i;

	if ((sv.state == ss_cinematic) ||
		(sv.state == ss_demo) ||
		(sv.state == ss_pic))
	{
		return;
	}

	/* send a message to each inactive client if needed */
	for (i = 0, c = svs.clients; i < maxclients->value; i++, c++)
	{
		if ((c->state == cs_free) || (c->state == cs_spawned))
		{
			continue;
		}

		if (c->netchan.message.overflowed)
		{
			SV_SendDisconnect(c);
			continue;
		}

		/* just update reliable	if needed */
		if (c->netchan.message.cursize ||
			(curtime - c->netchan.last_sent > 1000))
		{
			Netchan_Transmit(&c->netchan, 0, NULL);
		}
	}
}
