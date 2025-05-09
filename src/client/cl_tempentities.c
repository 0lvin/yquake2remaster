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
 * This file implements all temporary (dynamic created) entities
 *
 * =======================================================================
 */

#include "header/client.h"
#include "sound/header/local.h"

typedef enum
{
	ex_free, ex_explosion, ex_misc, ex_flash, ex_mflash, ex_poly, ex_poly2
} exptype_t;

typedef struct
{
	exptype_t type;
	entity_t ent;

	int frames;
	float light;
	vec3_t lightcolor;
	float start;
	int baseframe;
} explosion_t;

#define MAX_EXPLOSIONS 64
#define MAX_BEAMS 64
#define MAX_LASERS 64

explosion_t cl_explosions[MAX_EXPLOSIONS];

typedef struct
{
	int entity;
	int dest_entity;
	struct model_s *model;
	int endtime;
	vec3_t offset;
	vec3_t start, end;
} beam_t;

beam_t cl_beams[MAX_BEAMS];
beam_t cl_playerbeams[MAX_BEAMS];

typedef struct
{
	entity_t ent;
	int endtime;
} laser_t;
laser_t cl_lasers[MAX_LASERS];

cl_sustain_t cl_sustains[MAX_SUSTAINS];

extern void CL_TeleportParticles(vec3_t org);
void CL_BlasterParticles(vec3_t org, vec3_t dir);
void CL_BFGExplosionParticles(vec3_t org);
void CL_BlueBlasterParticles(vec3_t org, vec3_t dir);

void CL_ExplosionParticles(vec3_t org);
void CL_Explosion_Particle(vec3_t org, float size,
		qboolean large, qboolean rocket);

#define EXPLOSION_PARTICLES(x) CL_ExplosionParticles((x));

struct sfx_s *cl_sfx_ric1;
struct sfx_s *cl_sfx_ric2;
struct sfx_s *cl_sfx_ric3;
struct sfx_s *cl_sfx_lashit;
struct sfx_s *cl_sfx_spark5;
struct sfx_s *cl_sfx_spark6;
struct sfx_s *cl_sfx_spark7;
struct sfx_s *cl_sfx_railg;
struct sfx_s *cl_sfx_rockexp;
struct sfx_s *cl_sfx_grenexp;
struct sfx_s *cl_sfx_watrexp;
struct sfx_s *cl_sfx_plasexp;
struct sfx_s *cl_sfx_footsteps[4];

struct model_s *cl_mod_explode;
struct model_s *cl_mod_smoke;
struct model_s *cl_mod_flash;
struct model_s *cl_mod_parasite_segment;
struct model_s *cl_mod_grapple_cable;
struct model_s *cl_mod_parasite_tip;
struct model_s *cl_mod_explo4;
struct model_s *cl_mod_bfg_explo;
struct model_s *cl_mod_powerscreen;
struct model_s *cl_mod_plasmaexplo;

struct sfx_s *cl_sfx_lightning;
struct sfx_s *cl_sfx_disrexp;
struct model_s *cl_mod_lightning;
struct model_s *cl_mod_heatbeam;
struct model_s *cl_mod_monster_heatbeam;
struct model_s *cl_mod_explo4_big;

void
CL_RegisterTEntSounds(void)
{
	int i;

	cl_sfx_ric1 = S_RegisterSound("world/ric1.wav");
	cl_sfx_ric2 = S_RegisterSound("world/ric2.wav");
	cl_sfx_ric3 = S_RegisterSound("world/ric3.wav");
	cl_sfx_lashit = S_RegisterSound("weapons/lashit.wav");
	cl_sfx_spark5 = S_RegisterSound("world/spark5.wav");
	cl_sfx_spark6 = S_RegisterSound("world/spark6.wav");
	cl_sfx_spark7 = S_RegisterSound("world/spark7.wav");
	cl_sfx_railg = S_RegisterSound("weapons/railgf1a.wav");
	cl_sfx_rockexp = S_RegisterSound("weapons/rocklx1a.wav");
	cl_sfx_grenexp = S_RegisterSound("weapons/grenlx1a.wav");
	cl_sfx_watrexp = S_RegisterSound("weapons/xpld_wat.wav");
	S_RegisterSound("player/land1.wav");

	S_RegisterSound("player/fall2.wav");
	S_RegisterSound("player/fall1.wav");

	for (i = 0; i < 4; i++)
	{
		char name[MAX_QPATH];

		Com_sprintf(name, sizeof(name), "player/step%i.wav", i + 1);
		cl_sfx_footsteps[i] = S_RegisterSound(name);
	}

	cl_sfx_lightning = S_RegisterSound("weapons/tesla.wav");
	cl_sfx_disrexp = S_RegisterSound("weapons/disrupthit.wav");
}

void
CL_RegisterTEntModels(void)
{
	cl_mod_explode = R_RegisterModel("models/objects/explode/tris.md2");
	cl_mod_smoke = R_RegisterModel("models/objects/smoke/tris.md2");
	cl_mod_flash = R_RegisterModel("models/objects/flash/tris.md2");
	cl_mod_parasite_segment = R_RegisterModel("models/monsters/parasite/segment/tris.md2");
	cl_mod_grapple_cable = R_RegisterModel("models/ctf/segment/tris.md2");
	cl_mod_parasite_tip = R_RegisterModel("models/monsters/parasite/tip/tris.md2");
	cl_mod_explo4 = R_RegisterModel("models/objects/r_explode/tris.md2");
	cl_mod_bfg_explo = R_RegisterModel("sprites/s_bfg2.sp2");
	cl_mod_powerscreen = R_RegisterModel("models/items/armor/effect/tris.md2");

	R_RegisterModel("models/objects/laser/tris.md2");
	R_RegisterModel("models/objects/grenade2/tris.md2");
	R_RegisterModel("models/weapons/v_machn/tris.md2");
	R_RegisterModel("models/weapons/v_handgr/tris.md2");
	R_RegisterModel("models/weapons/v_shotg2/tris.md2");
	R_RegisterModel("models/objects/gibs/bone/tris.md2");
	R_RegisterModel("models/objects/gibs/sm_meat/tris.md2");
	R_RegisterModel("models/objects/gibs/bone2/tris.md2");

	Draw_FindPic("w_machinegun");
	Draw_FindPic("a_bullets");
	Draw_FindPic("i_health");
	Draw_FindPic("a_grenades");

	cl_mod_explo4_big = R_RegisterModel("models/objects/r_explode2/tris.md2");
	cl_mod_lightning = R_RegisterModel("models/proj/lightning/tris.md2");
	cl_mod_heatbeam = R_RegisterModel("models/proj/beam/tris.md2");
	cl_mod_monster_heatbeam = R_RegisterModel("models/proj/widowbeam/tris.md2");
}

void
CL_ClearTEnts(void)
{
	memset(cl_beams, 0, sizeof(cl_beams));
	memset(cl_explosions, 0, sizeof(cl_explosions));
	memset(cl_lasers, 0, sizeof(cl_lasers));

	memset(cl_playerbeams, 0, sizeof(cl_playerbeams));
	memset(cl_sustains, 0, sizeof(cl_sustains));
}

explosion_t *
CL_AllocExplosion(void)
{
	int i;
	float time;
	int index;

	for (i = 0; i < MAX_EXPLOSIONS; i++)
	{
		if (cl_explosions[i].type == ex_free)
		{
			memset(&cl_explosions[i], 0, sizeof(cl_explosions[i]));
			return &cl_explosions[i];
		}
	}

	/* find the oldest explosion */
	time = (float)cl.time;
	index = 0;

	for (i = 0; i < MAX_EXPLOSIONS; i++)
	{
		if (cl_explosions[i].start < time)
		{
			time = cl_explosions[i].start;
			index = i;
		}
	}

	memset(&cl_explosions[index], 0, sizeof(cl_explosions[index]));
	return &cl_explosions[index];
}

void
CL_SmokeAndFlash(vec3_t origin)
{
	explosion_t *ex;

	ex = CL_AllocExplosion();
	VectorCopy(origin, ex->ent.origin);
	ex->type = ex_misc;
	ex->frames = 4;
	ex->ent.flags = RF_TRANSLUCENT;
	ex->start = cl.frame.servertime - 100.0f;
	ex->ent.model = cl_mod_smoke;

	ex = CL_AllocExplosion();
	VectorCopy(origin, ex->ent.origin);
	ex->type = ex_flash;
	ex->ent.flags = RF_FULLBRIGHT;
	ex->frames = 2;
	ex->start = cl.frame.servertime - 100.0f;
	ex->ent.model = cl_mod_flash;
}

void
CL_ParseParticles(void)
{
	int color, count;
	vec3_t pos, dir;

	MSG_ReadPos(&net_message, pos, cls.serverProtocol);
	MSG_ReadDir(&net_message, dir);

	color = MSG_ReadByte(&net_message);

	count = MSG_ReadByte(&net_message);

	CL_ParticleEffect(pos, dir,
		VID_PaletteColor(color), VID_PaletteColor(color + 7), count);
}

void
CL_ParseBeam(struct model_s *model)
{
	int ent;
	vec3_t start, end;
	beam_t *b;
	int i;

	ent = MSG_ReadShort(&net_message);

	MSG_ReadPos(&net_message, start, cls.serverProtocol);
	MSG_ReadPos(&net_message, end, cls.serverProtocol);

	/* override any beam with the same entity */
	for (i = 0, b = cl_beams; i < MAX_BEAMS; i++, b++)
	{
		if (b->entity == ent)
		{
			b->entity = ent;
			b->model = model;
			b->endtime = cl.time + 200;
			VectorCopy(start, b->start);
			VectorCopy(end, b->end);
			VectorClear(b->offset);
			return;
		}
	}

	/* find a free beam */
	for (i = 0, b = cl_beams; i < MAX_BEAMS; i++, b++)
	{
		if (!b->model || (b->endtime < cl.time))
		{
			b->entity = ent;
			b->model = model;
			b->endtime = cl.time + 200;
			VectorCopy(start, b->start);
			VectorCopy(end, b->end);
			VectorClear(b->offset);
			return;
		}
	}

	Com_Printf("beam list overflow!\n");
	return;
}

void
CL_ParseBeam2(struct model_s *model)
{
	int ent;
	vec3_t start, end, offset;
	beam_t *b;
	int i;

	ent = MSG_ReadShort(&net_message);

	MSG_ReadPos(&net_message, start, cls.serverProtocol);
	MSG_ReadPos(&net_message, end, cls.serverProtocol);
	MSG_ReadPos(&net_message, offset, cls.serverProtocol);

	/* override any beam with the same entity */
	for (i = 0, b = cl_beams; i < MAX_BEAMS; i++, b++)
	{
		if (b->entity == ent)
		{
			b->entity = ent;
			b->model = model;
			b->endtime = cl.time + 200;
			VectorCopy(start, b->start);
			VectorCopy(end, b->end);
			VectorCopy(offset, b->offset);
			return;
		}
	}

	/* find a free beam */
	for (i = 0, b = cl_beams; i < MAX_BEAMS; i++, b++)
	{
		if (!b->model || (b->endtime < cl.time))
		{
			b->entity = ent;
			b->model = model;
			b->endtime = cl.time + 200;
			VectorCopy(start, b->start);
			VectorCopy(end, b->end);
			VectorCopy(offset, b->offset);
			return;
		}
	}

	Com_Printf("beam list overflow!\n");
	return;
}

/*
 * adds to the cl_playerbeam array instead of the cl_beams array
 */
void
CL_ParsePlayerBeam(struct model_s *model)
{
	int ent;
	vec3_t start, end, offset;
	beam_t *b;
	int i;

	ent = MSG_ReadShort(&net_message);

	MSG_ReadPos(&net_message, start, cls.serverProtocol);
	MSG_ReadPos(&net_message, end, cls.serverProtocol);

	/* network optimization */
	if (model == cl_mod_heatbeam)
	{
		VectorSet(offset, 2, 7, -3);
	}

	else if (model == cl_mod_monster_heatbeam)
	{
		model = cl_mod_heatbeam;
		VectorSet(offset, 0, 0, 0);
	}
	else
	{
		MSG_ReadPos(&net_message, offset, cls.serverProtocol);
	}

	/* Override any beam with the same entity
	   For player beams, we only want one per
	   player (entity) so... */
	for (i = 0, b = cl_playerbeams; i < MAX_BEAMS; i++, b++)
	{
		if (b->entity == ent)
		{
			b->entity = ent;
			b->model = model;
			b->endtime = cl.time + 200;
			VectorCopy(start, b->start);
			VectorCopy(end, b->end);
			VectorCopy(offset, b->offset);
			return;
		}
	}

	/* find a free beam */
	for (i = 0, b = cl_playerbeams; i < MAX_BEAMS; i++, b++)
	{
		if (!b->model || (b->endtime < cl.time))
		{
			b->entity = ent;
			b->model = model;
			b->endtime = cl.time + 100; /* this needs to be 100 to
										   prevent multiple heatbeams */
			VectorCopy(start, b->start);
			VectorCopy(end, b->end);
			VectorCopy(offset, b->offset);
			return;
		}
	}

	Com_Printf("beam list overflow!\n");
	return;
}

int
CL_ParseLightning(struct model_s *model)
{
	int srcEnt, destEnt;
	vec3_t start, end;
	beam_t *b;
	int i;

	srcEnt = MSG_ReadShort(&net_message);
	destEnt = MSG_ReadShort(&net_message);

	MSG_ReadPos(&net_message, start, cls.serverProtocol);
	MSG_ReadPos(&net_message, end, cls.serverProtocol);

	/* override any beam with the same
	   source AND destination entities */
	for (i = 0, b = cl_beams; i < MAX_BEAMS; i++, b++)
	{
		if ((b->entity == srcEnt) && (b->dest_entity == destEnt))
		{
			b->entity = srcEnt;
			b->dest_entity = destEnt;
			b->model = model;
			b->endtime = cl.time + 200;
			VectorCopy(start, b->start);
			VectorCopy(end, b->end);
			VectorClear(b->offset);
			return srcEnt;
		}
	}

	/* find a free beam */
	for (i = 0, b = cl_beams; i < MAX_BEAMS; i++, b++)
	{
		if (!b->model || (b->endtime < cl.time))
		{
			b->entity = srcEnt;
			b->dest_entity = destEnt;
			b->model = model;
			b->endtime = cl.time + 200;
			VectorCopy(start, b->start);
			VectorCopy(end, b->end);
			VectorClear(b->offset);
			return srcEnt;
		}
	}

	Com_Printf("beam list overflow!\n");
	return srcEnt;
}

void
CL_ParseLaser(int colors)
{
	vec3_t start;
	vec3_t end;
	laser_t *l;
	int i;

	MSG_ReadPos(&net_message, start, cls.serverProtocol);
	MSG_ReadPos(&net_message, end, cls.serverProtocol);

	for (i = 0, l = cl_lasers; i < MAX_LASERS; i++, l++)
	{
		if (l->endtime < cl.time)
		{
			float alpha = cl_laseralpha->value;
			if (alpha < 0.0f)
			{
				alpha = 0.0f;
			}
			else if (alpha > 1.0f)
			{
				alpha = 1.0f;
			}

			l->ent.flags = RF_TRANSLUCENT | RF_BEAM;
			VectorCopy(start, l->ent.origin);
			VectorCopy(end, l->ent.oldorigin);
			l->ent.alpha = alpha;
			l->ent.skinnum = (colors >> ((randk() % 4) * 8)) & 0xff;
			l->ent.model = NULL;
			l->ent.frame = 4;
			l->endtime = cl.time + 100;
			return;
		}
	}
}

void
CL_ParseSteam(void)
{
	vec3_t pos, dir;
	int id, i;
	int r;
	int cnt;
	int color;
	int magnitude;
	cl_sustain_t *s, *free_sustain;

	id = MSG_ReadShort(&net_message); /* an id of -1 is an instant effect */

	if (id != -1) /* sustains */
	{
		free_sustain = NULL;

		for (i = 0, s = cl_sustains; i < MAX_SUSTAINS; i++, s++)
		{
			if (s->id == 0)
			{
				free_sustain = s;
				break;
			}
		}

		if (free_sustain)
		{
			s->id = id;
			s->count = MSG_ReadByte(&net_message);
			MSG_ReadPos(&net_message, s->org, cls.serverProtocol);
			MSG_ReadDir(&net_message, s->dir);
			r = MSG_ReadByte(&net_message);
			s->basecolor = VID_PaletteColor(r & 0xff);
			s->finalcolor = VID_PaletteColor((r + 7) & 0xff);
			s->magnitude = MSG_ReadShort(&net_message);
			s->endtime = cl.time + MSG_ReadLong(&net_message);
			s->think = CL_ParticleSteamEffect2;
			s->thinkinterval = 100;
			s->nextthink = cl.time;
		}
		else
		{
			MSG_ReadByte(&net_message);
			MSG_ReadPos(&net_message, pos, cls.serverProtocol);
			MSG_ReadDir(&net_message, dir);
			MSG_ReadByte(&net_message);
			MSG_ReadShort(&net_message);
			MSG_ReadLong(&net_message); /* really interval */
		}
	}
	else
	{
		/* instant */
		cnt = MSG_ReadByte(&net_message);
		MSG_ReadPos(&net_message, pos, cls.serverProtocol);
		MSG_ReadDir(&net_message, dir);
		r = MSG_ReadByte(&net_message);
		magnitude = MSG_ReadShort(&net_message);
		color = r & 0xff;
		CL_ParticleSteamEffect(pos, dir,
			VID_PaletteColor(color), VID_PaletteColor(color + 7), cnt, magnitude);
	}
}

void
CL_ParseWidow(void)
{
	vec3_t pos;
	int id, i;
	cl_sustain_t *s, *free_sustain;

	id = MSG_ReadShort(&net_message);

	free_sustain = NULL;

	for (i = 0, s = cl_sustains; i < MAX_SUSTAINS; i++, s++)
	{
		if (s->id == 0)
		{
			free_sustain = s;
			break;
		}
	}

	if (free_sustain)
	{
		s->id = id;
		MSG_ReadPos(&net_message, s->org, cls.serverProtocol);
		s->endtime = cl.time + 2100;
		s->think = CL_Widowbeamout;
		s->thinkinterval = 1;
		s->nextthink = cl.time;
	}
	else
	{
		/* no free sustains */
		MSG_ReadPos(&net_message, pos, cls.serverProtocol);
	}
}

void
CL_ParseNuke(void)
{
	vec3_t pos;
	int i;
	cl_sustain_t *s, *free_sustain;

	free_sustain = NULL;

	for (i = 0, s = cl_sustains; i < MAX_SUSTAINS; i++, s++)
	{
		if (s->id == 0)
		{
			free_sustain = s;
			break;
		}
	}

	if (free_sustain)
	{
		s->id = 21000;
		MSG_ReadPos(&net_message, s->org, cls.serverProtocol);
		s->endtime = cl.time + 1000;
		s->think = CL_Nukeblast;
		s->thinkinterval = 1;
		s->nextthink = cl.time;
	}
	else
	{
		/* no free sustains */
		MSG_ReadPos(&net_message, pos, cls.serverProtocol);
	}
}

static unsigned int splash_color[] = {
	0xff000000, 0xff6b6b6b,
	0xff07abff, 0xff002bab,
	0xffcf7b77, 0xff734747,
	0xff4b5f7b, 0xff2b374b,
	0xff00ff00, 0xffffffff,
	0xff07abff, 0xff002bab,
	0xff001f9b, 0xff00001b,
};

void
CL_ParseTEnt(void)
{
	temp_event_t type;
	vec3_t pos, pos2, dir;
	explosion_t *ex;
	int cnt;
	int color;
	int r;
	int ent;
	int magnitude;

	type = MSG_ReadByte(&net_message);

	switch (type)
	{
		case TE_BLOOD: /* bullet hitting flesh */
			MSG_ReadPos(&net_message, pos, cls.serverProtocol);
			MSG_ReadDir(&net_message, dir);
			CL_ParticleEffect(pos, dir, 0xff001f9b, 0xff00001b, 60);
			break;

		case TE_GUNSHOT: /* bullet hitting wall */
		case TE_SPARKS:
		case TE_BULLET_SPARKS:
			MSG_ReadPos(&net_message, pos, cls.serverProtocol);
			MSG_ReadDir(&net_message, dir);

			if (type == TE_GUNSHOT)
			{
				CL_ParticleEffect(pos, dir, 0xff000000, 0xff6b6b6b, 40);
			}
			else
			{
				CL_ParticleEffect(pos, dir, 0xff07abff, 0xff002bab, 6);
			}

			if (type != TE_SPARKS)
			{
				CL_SmokeAndFlash(pos);
				/* impact sound */
				cnt = randk() & 15;

				if (cnt == 1)
				{
					S_StartSound(pos, 0, 0, cl_sfx_ric1, 1, ATTN_NORM, 0);
				}
				else if (cnt == 2)
				{
					S_StartSound(pos, 0, 0, cl_sfx_ric2, 1, ATTN_NORM, 0);
				}
				else if (cnt == 3)
				{
					S_StartSound(pos, 0, 0, cl_sfx_ric3, 1, ATTN_NORM, 0);
				}
			}

			break;

		case TE_SCREEN_SPARKS:
		case TE_SHIELD_SPARKS:
			MSG_ReadPos(&net_message, pos, cls.serverProtocol);
			MSG_ReadDir(&net_message, dir);

			if (type == TE_SCREEN_SPARKS)
			{
				CL_ParticleEffect(pos, dir, 0xff00ff00, 0xffffffff, 40);
			}

			else
			{
				CL_ParticleEffect(pos, dir, 0xffcf7b77, 0xff734747, 40);
			}

			if (cl_limitsparksounds->value)
			{
				num_power_sounds++;

				/* If too many of these sounds are started in one frame
				 * (for example if the player shoots with the super
				 * shotgun into the power screen of a Brain) things get
				 * too loud and OpenAL is forced to scale the volume of
				 * several other sounds and the background music down.
				 * That leads to a noticable and annoying drop in the
				 * overall volume.
				 *
				 * Work around that by limiting the number of sounds
				 * started.
				 * 16 was choosen by empirical testing.
				 *
				 * This was fixed in openal-soft 0.19.0. We're keeping
				 * the work around hidden behind a cvar and no longer
				 * limited to OpenAL because a) some Linux distros may
				 * still ship older openal-soft versions and b) some
				 * player may like the changed behavior.
				 */
				if (num_power_sounds < 16)
				{
					S_StartSound(pos, 0, 0, cl_sfx_lashit, 1, ATTN_NORM, 0);
				}
			}
			else
			{
				S_StartSound(pos, 0, 0, cl_sfx_lashit, 1, ATTN_NORM, 0);
			}

			break;

		case TE_SHOTGUN: /* bullet hitting wall */
			MSG_ReadPos(&net_message, pos, cls.serverProtocol);
			MSG_ReadDir(&net_message, dir);
			CL_ParticleEffect(pos, dir, 0xff000000, 0xff6b6b6b, 20);
			CL_SmokeAndFlash(pos);
			break;

		case TE_SPLASH: /* bullet hitting water */
			cnt = MSG_ReadByte(&net_message);
			MSG_ReadPos(&net_message, pos, cls.serverProtocol);
			MSG_ReadDir(&net_message, dir);
			r = MSG_ReadByte(&net_message);

			if (r > 6)
			{
				r = 0;
			}

			CL_ParticleEffect(pos, dir,
				splash_color[r * 2], splash_color[r * 2 + 1], cnt);

			if (r == SPLASH_SPARKS)
			{
				r = randk() & 3;

				if (r == 0)
				{
					S_StartSound(pos, 0, 0, cl_sfx_spark5, 1, ATTN_STATIC, 0);
				}
				else if (r == 1)
				{
					S_StartSound(pos, 0, 0, cl_sfx_spark6, 1, ATTN_STATIC, 0);
				}
				else
				{
					S_StartSound(pos, 0, 0, cl_sfx_spark7, 1, ATTN_STATIC, 0);
				}
			}

			break;

		case TE_LASER_SPARKS:
			cnt = MSG_ReadByte(&net_message);
			MSG_ReadPos(&net_message, pos, cls.serverProtocol);
			MSG_ReadDir(&net_message, dir);
			color = MSG_ReadByte(&net_message);
			CL_ParticleEffect2(pos, dir,
				VID_PaletteColor(color), VID_PaletteColor(color + 7), cnt);
			break;

		case TE_BLUEHYPERBLASTER:
			MSG_ReadPos(&net_message, pos, cls.serverProtocol);
			MSG_ReadPos(&net_message, dir, cls.serverProtocol);
			CL_BlasterParticles(pos, dir);
			break;

		case TE_BLASTER: /* blaster hitting wall */
			MSG_ReadPos(&net_message, pos, cls.serverProtocol);
			MSG_ReadDir(&net_message, dir);
			CL_BlasterParticles(pos, dir);

			ex = CL_AllocExplosion();
			VectorCopy(pos, ex->ent.origin);
			ex->ent.angles[0] = (float)acos(dir[2]) / M_PI * 180;

			if (dir[0])
			{
				ex->ent.angles[1] = (float)atan2(dir[1], dir[0]) / M_PI * 180;
			}

			else if (dir[1] > 0)
			{
				ex->ent.angles[1] = 90;
			}
			else if (dir[1] < 0)
			{
				ex->ent.angles[1] = 270;
			}
			else
			{
				ex->ent.angles[1] = 0;
			}

			ex->type = ex_misc;
			ex->ent.flags = 0;
			ex->start = cl.frame.servertime - 100.0f;
			ex->light = 150;
			ex->lightcolor[0] = 1;
			ex->lightcolor[1] = 1;
			ex->ent.model = cl_mod_explode;
			ex->frames = 4;
			S_StartSound(pos, 0, 0, cl_sfx_lashit, 1, ATTN_NORM, 0);
			break;

		case TE_RAILTRAIL: /* railgun effect */
		case TE_RAILTRAIL2:
			MSG_ReadPos(&net_message, pos, cls.serverProtocol);
			MSG_ReadPos(&net_message, pos2, cls.serverProtocol);
			CL_RailTrail(pos, pos2);
			S_StartSound(pos2, 0, 0, cl_sfx_railg, 1, ATTN_NORM, 0);
			break;

		case TE_EXPLOSION2:
		case TE_GRENADE_EXPLOSION:
		case TE_GRENADE_EXPLOSION_WATER:
			MSG_ReadPos(&net_message, pos, cls.serverProtocol);
			ex = CL_AllocExplosion();
			VectorCopy(pos, ex->ent.origin);
			ex->type = ex_poly;
			ex->ent.flags = RF_FULLBRIGHT | RF_NOSHADOW;
			ex->start = cl.frame.servertime - 100.0f;
			ex->light = 350;
			ex->lightcolor[0] = 1.0;
			ex->lightcolor[1] = 0.5;
			ex->lightcolor[2] = 0.5;
			ex->ent.model = cl_mod_explo4;
			ex->frames = 19;
			ex->baseframe = 30;
			ex->ent.angles[1] = (float)(randk() % 360);
			EXPLOSION_PARTICLES(pos);

			if (type == TE_GRENADE_EXPLOSION_WATER)
			{
				S_StartSound(pos, 0, 0, cl_sfx_watrexp, 1, ATTN_NORM, 0);
			}
			else
			{
				S_StartSound(pos, 0, 0, cl_sfx_grenexp, 1, ATTN_NORM, 0);
			}

			break;

		case TE_PLASMA_EXPLOSION:
			MSG_ReadPos(&net_message, pos, cls.serverProtocol);
			ex = CL_AllocExplosion();
			VectorCopy(pos, ex->ent.origin);
			ex->type = ex_poly;
			ex->ent.flags = RF_FULLBRIGHT | RF_NOSHADOW;
			ex->start = cl.frame.servertime - 100.0f;
			ex->light = 350;
			ex->lightcolor[0] = 1.0;
			ex->lightcolor[1] = 0.5;
			ex->lightcolor[2] = 0.5;
			ex->ent.angles[1] = (float)(randk() % 360);
			ex->ent.model = cl_mod_explo4;

			if (frandk() < 0.5)
			{
				ex->baseframe = 15;
			}

			ex->frames = 15;
			EXPLOSION_PARTICLES(pos);
			S_StartSound(pos, 0, 0, cl_sfx_rockexp, 1, ATTN_NORM, 0);
			break;

		case TE_EXPLOSION1_BIG:
		case TE_EXPLOSION1_NP:
		case TE_EXPLOSION1:
		case TE_ROCKET_EXPLOSION:
		case TE_ROCKET_EXPLOSION_WATER:
			MSG_ReadPos(&net_message, pos, cls.serverProtocol);
			ex = CL_AllocExplosion();
			VectorCopy(pos, ex->ent.origin);
			ex->type = ex_poly;
			ex->ent.flags = RF_FULLBRIGHT | RF_NOSHADOW;
			ex->start = cl.frame.servertime - 100.0f;
			ex->light = 350;
			ex->lightcolor[0] = 1.0;
			ex->lightcolor[1] = 0.5;
			ex->lightcolor[2] = 0.5;
			ex->ent.angles[1] = (float)(randk() % 360);

			if (type != TE_EXPLOSION1_BIG)
			{
				ex->ent.model = cl_mod_explo4;
			}
			else
			{
				ex->ent.model = cl_mod_explo4_big;
			}

			if (frandk() < 0.5)
			{
				ex->baseframe = 15;
			}

			ex->frames = 15;

			if ((type != TE_EXPLOSION1_BIG) && (type != TE_EXPLOSION1_NP))
			{
				EXPLOSION_PARTICLES(pos);
			}

			if (type == TE_ROCKET_EXPLOSION_WATER)
			{
				S_StartSound(pos, 0, 0, cl_sfx_watrexp, 1, ATTN_NORM, 0);
			}
			else
			{
				S_StartSound(pos, 0, 0, cl_sfx_rockexp, 1, ATTN_NORM, 0);
			}

			break;

		case TE_BFG_EXPLOSION:
			MSG_ReadPos(&net_message, pos, cls.serverProtocol);
			ex = CL_AllocExplosion();
			VectorCopy(pos, ex->ent.origin);
			ex->type = ex_poly;
			ex->ent.flags = RF_FULLBRIGHT | RF_NOSHADOW;
			ex->start = cl.frame.servertime - 100.0f;
			ex->light = 350;
			ex->lightcolor[0] = 0.0;
			ex->lightcolor[1] = 1.0;
			ex->lightcolor[2] = 0.0;
			ex->ent.model = cl_mod_bfg_explo;
			ex->ent.flags |= RF_TRANSLUCENT;
			ex->ent.alpha = 0.30f;
			ex->frames = 4;
			break;

		case TE_BFG_BIGEXPLOSION:
			MSG_ReadPos(&net_message, pos, cls.serverProtocol);
			CL_BFGExplosionParticles(pos);
			break;

		case TE_BFG_LASER:
			CL_ParseLaser(0xd0d1d2d3);
			break;

		case TE_BUBBLETRAIL:
			MSG_ReadPos(&net_message, pos, cls.serverProtocol);
			MSG_ReadPos(&net_message, pos2, cls.serverProtocol);
			CL_BubbleTrail(pos, pos2);
			break;

		case TE_PARASITE_ATTACK:
		case TE_MEDIC_CABLE_ATTACK:
			CL_ParseBeam(cl_mod_parasite_segment);
			break;

		case TE_BOSSTPORT: /* boss teleporting to station */
			MSG_ReadPos(&net_message, pos, cls.serverProtocol);
			CL_BigTeleportParticles(pos);
			S_StartSound(pos, 0, 0, S_RegisterSound(
						"misc/bigtele.wav"), 1, ATTN_NONE, 0);
			break;

		case TE_GRAPPLE_CABLE:
			CL_ParseBeam2(cl_mod_grapple_cable);
			break;

		case TE_WELDING_SPARKS:
			cnt = MSG_ReadByte(&net_message);
			MSG_ReadPos(&net_message, pos, cls.serverProtocol);
			MSG_ReadDir(&net_message, dir);
			color = MSG_ReadByte(&net_message);
			CL_ParticleEffect2(pos, dir,
				VID_PaletteColor(color), VID_PaletteColor(color + 7), cnt);

			ex = CL_AllocExplosion();
			VectorCopy(pos, ex->ent.origin);
			ex->type = ex_flash;
			ex->ent.flags = RF_BEAM;
			ex->start = cl.frame.servertime - 0.1f;
			ex->light = 100 + (float)(randk() % 75);
			ex->lightcolor[0] = 1.0f;
			ex->lightcolor[1] = 1.0f;
			ex->lightcolor[2] = 0.3f;
			ex->ent.model = cl_mod_flash;
			ex->frames = 2;
			break;

		case TE_GREENBLOOD:
			MSG_ReadPos(&net_message, pos, cls.serverProtocol);
			MSG_ReadDir(&net_message, dir);
			CL_ParticleEffect2(pos, dir, 0xff0fbfff, 0xff003bb7, 30);
			break;

		case TE_TUNNEL_SPARKS:
			cnt = MSG_ReadByte(&net_message);
			MSG_ReadPos(&net_message, pos, cls.serverProtocol);
			MSG_ReadDir(&net_message, dir);
			color = MSG_ReadByte(&net_message);
			CL_ParticleEffect3(pos, dir, VID_PaletteColor(color), cnt);
			break;

		case TE_BLASTER2:
		case TE_FLECHETTE:
			MSG_ReadPos(&net_message, pos, cls.serverProtocol);
			MSG_ReadDir(&net_message, dir);

			if (type == TE_BLASTER2)
			{
				CL_BlasterParticles2(pos, dir, 0xff00ff00, 0xffffffff);
			}
			else
			{
				CL_BlasterParticles2(pos, dir, 0xffb7a787, 0xff5b430f);
			}

			ex = CL_AllocExplosion();
			VectorCopy(pos, ex->ent.origin);
			ex->ent.angles[0] = (float)acos(dir[2]) / M_PI * 180;

			if (dir[0])
			{
				ex->ent.angles[1] = (float)atan2(dir[1], dir[0]) / M_PI * 180;
			}
			else if (dir[1] > 0)
			{
				ex->ent.angles[1] = 90;
			}

			else if (dir[1] < 0)
			{
				ex->ent.angles[1] = 270;
			}
			else
			{
				ex->ent.angles[1] = 0;
			}

			ex->type = ex_misc;
			ex->ent.flags = RF_FULLBRIGHT | RF_TRANSLUCENT;

			if (type == TE_BLASTER2)
			{
				ex->ent.skinnum = 1;
			}
			else /* flechette */
			{
				ex->ent.skinnum = 2;
			}

			ex->start = cl.frame.servertime - 100.0f;
			ex->light = 150;

			if (type == TE_BLASTER2)
			{
				ex->lightcolor[1] = 1;
			}
			else
			{
				/* flechette */
				ex->lightcolor[0] = 0.19f;
				ex->lightcolor[1] = 0.41f;
				ex->lightcolor[2] = 0.75f;
			}

			ex->ent.model = cl_mod_explode;
			ex->frames = 4;
			S_StartSound(pos, 0, 0, cl_sfx_lashit, 1, ATTN_NORM, 0);
			break;

		case TE_FLAME:
			MSG_ReadPos(&net_message, pos, cls.serverProtocol);
			CL_FlameEffects(pos);
			break;

		case TE_LIGHTNING:
			ent = CL_ParseLightning(cl_mod_lightning);
			S_StartSound(NULL, ent, CHAN_WEAPON, cl_sfx_lightning,
				1, ATTN_NORM, 0);
			break;

		case TE_DEBUGTRAIL:
			MSG_ReadPos(&net_message, pos, cls.serverProtocol);
			MSG_ReadPos(&net_message, pos2, cls.serverProtocol);
			CL_DebugTrail(pos, pos2);
			break;

		case TE_PLAIN_EXPLOSION:
			MSG_ReadPos(&net_message, pos, cls.serverProtocol);

			ex = CL_AllocExplosion();
			VectorCopy(pos, ex->ent.origin);
			ex->type = ex_poly;
			ex->ent.flags = RF_FULLBRIGHT | RF_NOSHADOW;
			ex->start = cl.frame.servertime - 100.0f;
			ex->light = 350;
			ex->lightcolor[0] = 1.0;
			ex->lightcolor[1] = 0.5;
			ex->lightcolor[2] = 0.5;
			ex->ent.angles[1] = randk() % 360;
			ex->ent.model = cl_mod_explo4;

			if (frandk() < 0.5)
			{
				ex->baseframe = 15;
			}

			ex->frames = 15;

			S_StartSound(pos, 0, 0, cl_sfx_rockexp, 1, ATTN_NORM, 0);

			break;

		case TE_FLASHLIGHT:
			MSG_ReadPos(&net_message, pos, cls.serverProtocol);
			ent = MSG_ReadShort(&net_message);
			CL_Flashlight(ent, pos);
			break;

		case TE_FORCEWALL:
			MSG_ReadPos(&net_message, pos, cls.serverProtocol);
			MSG_ReadPos(&net_message, pos2, cls.serverProtocol);
			color = MSG_ReadByte(&net_message);
			CL_ForceWall(pos, pos2, VID_PaletteColor(color));
			break;

		case TE_HEATBEAM:
			CL_ParsePlayerBeam(cl_mod_heatbeam);
			break;

		case TE_MONSTER_HEATBEAM:
			CL_ParsePlayerBeam(cl_mod_monster_heatbeam);
			break;

		case TE_HEATBEAM_SPARKS:
			cnt = 50;
			MSG_ReadPos(&net_message, pos, cls.serverProtocol);
			MSG_ReadDir(&net_message, dir);
			r = 8;
			magnitude = 60;
			CL_ParticleSteamEffect(pos, dir, 0xff7b7b7b, 0xffebebeb, cnt, magnitude);
			S_StartSound(pos, 0, 0, cl_sfx_lashit, 1, ATTN_NORM, 0);
			break;

		case TE_HEATBEAM_STEAM:
			cnt = 20;
			MSG_ReadPos(&net_message, pos, cls.serverProtocol);
			MSG_ReadDir(&net_message, dir);
			magnitude = 60;
			CL_ParticleSteamEffect(pos, dir, 0xff07abff, 0xff002bab, cnt, magnitude);
			S_StartSound(pos, 0, 0, cl_sfx_lashit, 1, ATTN_NORM, 0);
			break;

		case TE_STEAM:
			CL_ParseSteam();
			break;

		case TE_BUBBLETRAIL2:
			MSG_ReadPos(&net_message, pos, cls.serverProtocol);
			MSG_ReadPos(&net_message, pos2, cls.serverProtocol);
			CL_BubbleTrail2(pos, pos2, 8);
			S_StartSound(pos, 0, 0, cl_sfx_lashit, 1, ATTN_NORM, 0);
			break;

		case TE_MOREBLOOD:
			MSG_ReadPos(&net_message, pos, cls.serverProtocol);
			MSG_ReadDir(&net_message, dir);
			CL_ParticleEffect(pos, dir, 0xff001f9b, 0xff00001b, 250);
			break;

		case TE_CHAINFIST_SMOKE:
			dir[0] = 0;
			dir[1] = 0;
			dir[2] = 1;
			MSG_ReadPos(&net_message, pos, cls.serverProtocol);
			CL_ParticleSmokeEffect(pos, dir, 0xff000000, 0xff6b6b6b, 20, 20);
			break;

		case TE_ELECTRIC_SPARKS:
			MSG_ReadPos(&net_message, pos, cls.serverProtocol);
			MSG_ReadDir(&net_message, dir);
			CL_ParticleEffect(pos, dir, 0xff5b430f, 0xff1f1700, 40);
			S_StartSound(pos, 0, 0, cl_sfx_lashit, 1, ATTN_NORM, 0);
			break;

		case TE_TRACKER_EXPLOSION:
			MSG_ReadPos(&net_message, pos, cls.serverProtocol);
			CL_ColorFlash(pos, 0, 150, -1, -1, -1);
			CL_ColorExplosionParticles(pos, 0xff000000, 0xff0f0f0f);
			S_StartSound(pos, 0, 0, cl_sfx_disrexp, 1, ATTN_NORM, 0);
			break;

		case TE_TELEPORT_EFFECT:
		case TE_DBALL_GOAL:
			MSG_ReadPos(&net_message, pos, cls.serverProtocol);
			CL_TeleportParticles(pos);
			break;

		case TE_WIDOWBEAMOUT:
			CL_ParseWidow();
			break;

		case TE_NUKEBLAST:
			CL_ParseNuke();
			break;

		case TE_WIDOWSPLASH:
			MSG_ReadPos(&net_message, pos, cls.serverProtocol);
			CL_WidowSplash(pos);
			break;

		default:
			Com_Error(ERR_DROP, "%s: bad type", __func__);
	}
}

void
CL_AddBeams(void)
{
	int i, j;
	beam_t *b;
	vec3_t dist, org;
	float d;
	entity_t ent;
	float yaw, pitch;
	float forward;
	float len, steps;
	float model_length;

	/* update beams */
	for (i = 0, b = cl_beams; i < MAX_BEAMS; i++, b++)
	{
		if (!b->model || (b->endtime < cl.time))
		{
			continue;
		}

		/* if coming from the player, update the start position */
		if (b->entity == cl.playernum + 1) /* entity 0 is the world */
		{
			VectorCopy(cl.refdef.vieworg, b->start);
			b->start[2] -= 22; /* adjust for view height */
		}

		VectorAdd(b->start, b->offset, org);

		/* calculate pitch and yaw */
		VectorSubtract(b->end, org, dist);

		if ((dist[1] == 0) && (dist[0] == 0))
		{
			yaw = 0;

			if (dist[2] > 0)
			{
				pitch = 90;
			}

			else
			{
				pitch = 270;
			}
		}
		else
		{
			if (dist[0])
			{
				yaw = ((float)atan2(dist[1], dist[0]) * 180 / M_PI);
			}

			else if (dist[1] > 0)
			{
				yaw = 90;
			}

			else
			{
				yaw = 270;
			}

			if (yaw < 0)
			{
				yaw += 360;
			}

			forward = (float)sqrt(dist[0] * dist[0] + dist[1] * dist[1]);
			pitch = ((float)atan2(dist[2], forward) * -180.0 / M_PI);

			if (pitch < 0)
			{
				pitch += 360.0;
			}
		}

		/* add new entities for the beams */
		d = VectorNormalize(dist);

		memset(&ent, 0, sizeof(ent));

		if (b->model == cl_mod_lightning)
		{
			model_length = 35.0;
			d -= 20.0; /* correction so it doesn't end in middle of tesla */
		}
		else
		{
			model_length = 30.0;
		}

		steps = (float)ceil(d / model_length);
		len = (d - model_length) / (steps - 1);

		/* special case for lightning model .. if the real length
		   is shorter than the model, flip it around & draw it
		   from the end to the start. This prevents the model from
		   going through the tesla mine (instead it goes through
		   the target) */
		if ((b->model == cl_mod_lightning) && (d <= model_length))
		{
			VectorCopy(b->end, ent.origin);
			ent.model = b->model;
			ent.flags = RF_FULLBRIGHT;
			ent.angles[0] = pitch;
			ent.angles[1] = yaw;
			ent.angles[2] = (float)(randk() % 360);
			V_AddEntity(&ent);
			return;
		}

		while (d > 0)
		{
			VectorCopy(org, ent.origin);
			ent.model = b->model;

			if (b->model == cl_mod_lightning)
			{
				ent.flags = RF_FULLBRIGHT;
				ent.angles[0] = -pitch;
				ent.angles[1] = yaw + 180.0f;
				ent.angles[2] = (float)(randk() % 360);
			}
			else
			{
				ent.angles[0] = pitch;
				ent.angles[1] = yaw;
				ent.angles[2] = (float)(randk() % 360);
			}

			V_AddEntity(&ent);

			for (j = 0; j < 3; j++)
			{
				org[j] += dist[j] * len;
			}

			d -= model_length;
		}
	}
}

extern cvar_t *hand;

void
CL_AddPlayerBeams(void)
{
	int i, j;
	beam_t *b;
	vec3_t dist, org;
	float d;
	entity_t ent;
	float yaw, pitch;
	float forward;
	float len, steps;
	int framenum;
	float model_length;

	float hand_multiplier;
	frame_t *oldframe;
	player_state_t *ps, *ops;

	framenum = 0;

	if (hand)
	{
		if (hand->value == 2)
		{
			hand_multiplier = 0;
		}

		else if (hand->value == 1)
		{
			hand_multiplier = -1;
		}

		else
		{
			hand_multiplier = 1;
		}
	}
	else
	{
		hand_multiplier = 1;
	}

	/* update beams */
	for (i = 0, b = cl_playerbeams; i < MAX_BEAMS; i++, b++)
	{
		vec3_t f, r, u;

		if (!b->model || (b->endtime < cl.time))
		{
			continue;
		}

		if (cl_mod_heatbeam && (b->model == cl_mod_heatbeam))
		{
			/* if coming from the player, update the start position */
			if (b->entity == cl.playernum + 1)
			{
				/* set up gun position */
				ps = &cl.frame.playerstate;
				j = (cl.frame.serverframe - 1) & UPDATE_MASK;
				oldframe = &cl.frames[j];

				if ((oldframe->serverframe != cl.frame.serverframe - 1) || !oldframe->valid)
				{
					oldframe = &cl.frame; /* previous frame was dropped or invalid */
				}

				ops = &oldframe->playerstate;

				for (j = 0; j < 3; j++)
				{
					b->start[j] = cl.refdef.vieworg[j] + ops->gunoffset[j]
								  + cl.lerpfrac * (ps->gunoffset[j] - ops->gunoffset[j]);
				}

				VectorMA(b->start, (hand_multiplier * b->offset[0]),
						cl.v_right, org);
				VectorMA(org, b->offset[1], cl.v_forward, org);
				VectorMA(org, b->offset[2], cl.v_up, org);

				if ((hand) && (hand->value == 2))
				{
					VectorMA(org, -1, cl.v_up, org);
				}

				VectorCopy(cl.v_right, r);
				VectorCopy(cl.v_forward, f);
				VectorCopy(cl.v_up, u);
			}
			else
			{
				VectorCopy(b->start, org);
			}
		}
		else
		{
			/* if coming from the player, update the start position */
			if (b->entity == cl.playernum + 1) /* entity 0 is the world */
			{
				VectorCopy(cl.refdef.vieworg, b->start);
				b->start[2] -= 22; /* adjust for view height */
			}

			VectorAdd(b->start, b->offset, org);
		}

		/* calculate pitch and yaw */
		VectorSubtract(b->end, org, dist);

		if (cl_mod_heatbeam && (b->model == cl_mod_heatbeam) &&
			(b->entity == cl.playernum + 1))
		{
			vec_t len;

			len = VectorLength(dist);
			VectorScale(f, len, dist);
			VectorMA(dist, (hand_multiplier * b->offset[0]), r, dist);
			VectorMA(dist, b->offset[1], f, dist);
			VectorMA(dist, b->offset[2], u, dist);

			if ((hand) && (hand->value == 2))
			{
				VectorMA(org, -1, cl.v_up, org);
			}
		}

		if ((dist[1] == 0) && (dist[0] == 0))
		{
			yaw = 0;

			if (dist[2] > 0)
			{
				pitch = 90;
			}

			else
			{
				pitch = 270;
			}
		}
		else
		{
			if (dist[0])
			{
				yaw = ((float)atan2(dist[1], dist[0]) * 180 / M_PI);
			}

			else if (dist[1] > 0)
			{
				yaw = 90;
			}

			else
			{
				yaw = 270;
			}

			if (yaw < 0)
			{
				yaw += 360;
			}

			forward = sqrt(dist[0] * dist[0] + dist[1] * dist[1]);
			pitch = ((float)atan2(dist[2], forward) * -180.0 / M_PI);

			if (pitch < 0)
			{
				pitch += 360.0;
			}
		}

		if (cl_mod_heatbeam && (b->model == cl_mod_heatbeam))
		{
			if (b->entity != cl.playernum + 1)
			{
				framenum = 2;
				ent.angles[0] = -pitch;
				ent.angles[1] = yaw + 180.0f;
				ent.angles[2] = 0;
				AngleVectors(ent.angles, f, r, u);

				/* if it's a non-origin offset, it's a player, so use the hardcoded player offset */
				if (!VectorCompare(b->offset, vec3_origin))
				{
					VectorMA(org, -(b->offset[0]) + 1, r, org);
					VectorMA(org, -(b->offset[1]), f, org);
					VectorMA(org, -(b->offset[2]) - 10, u, org);
				}
				else
				{
					/* if it's a monster, do the particle effect */
					CL_MonsterPlasma_Shell(b->start);
				}
			}
			else
			{
				framenum = 1;
			}
		}

		/* if it's the heatbeam, draw the particle effect */
		if ((cl_mod_heatbeam && (b->model == cl_mod_heatbeam) &&
			 (b->entity == cl.playernum + 1)))
		{
			CL_Heatbeam(org, dist);
		}

		/* add new entities for the beams */
		d = VectorNormalize(dist);

		memset(&ent, 0, sizeof(ent));

		if (b->model == cl_mod_heatbeam)
		{
			model_length = 32.0;
		}
		else if (b->model == cl_mod_lightning)
		{
			model_length = 35.0;
			d -= 20.0; /* correction so it doesn't end in middle of tesla */
		}
		else
		{
			model_length = 30.0;
		}

		steps = ceil(d / model_length);
		len = (d - model_length) / (steps - 1);

		/* special case for lightning model .. if the real
		   length is shorter than the model, flip it around
		   & draw it from the end to the start. This prevents
		   the model from going through the tesla mine
		   (instead it goes through the target) */
		if ((b->model == cl_mod_lightning) && (d <= model_length))
		{
			VectorCopy(b->end, ent.origin);
			ent.model = b->model;
			ent.flags = RF_FULLBRIGHT;
			ent.angles[0] = pitch;
			ent.angles[1] = yaw;
			ent.angles[2] = (float)(randk() % 360);
			V_AddEntity(&ent);
			return;
		}

		while (d > 0)
		{
			VectorCopy(org, ent.origin);
			ent.model = b->model;

			if (cl_mod_heatbeam && (b->model == cl_mod_heatbeam))
			{
				ent.flags = RF_FULLBRIGHT|RF_WEAPONMODEL; // DG: fix rogue heatbeam high FOV rendering
				ent.angles[0] = -pitch;
				ent.angles[1] = yaw + 180.0f;
				ent.angles[2] = (float)((cl.time) % 360);
				ent.frame = framenum;
			}
			else if (b->model == cl_mod_lightning)
			{
				ent.flags = RF_FULLBRIGHT;
				ent.angles[0] = -pitch;
				ent.angles[1] = yaw + 180.0f;
				ent.angles[2] = (float)(randk() % 360);
			}
			else
			{
				ent.angles[0] = pitch;
				ent.angles[1] = yaw;
				ent.angles[2] = (float)(randk() % 360);
			}

			V_AddEntity(&ent);

			for (j = 0; j < 3; j++)
			{
				org[j] += dist[j] * len;
			}

			d -= model_length;
		}
	}
}

void
CL_AddExplosions(void)
{
	entity_t *ent;
	int i;
	explosion_t *ex;
	float frac;
	int f;

	memset(&ent, 0, sizeof(ent));

	for (i = 0, ex = cl_explosions; i < MAX_EXPLOSIONS; i++, ex++)
	{
		if (ex->type == ex_free)
		{
			continue;
		}

		frac = (cl.time - ex->start) / 100.0;
		f = (int)floor(frac);

		ent = &ex->ent;

		switch (ex->type)
		{
			case ex_mflash:

				if (f >= ex->frames - 1)
				{
					ex->type = ex_free;
				}

				break;
			case ex_misc:

				if (f >= ex->frames - 1)
				{
					ex->type = ex_free;
					break;
				}

				ent->alpha = 1.0f - frac / (ex->frames - 1);
				break;
			case ex_flash:

				if (f >= 1)
				{
					ex->type = ex_free;
					break;
				}

				ent->alpha = 1.0;
				break;
			case ex_poly:

				if (f >= ex->frames - 1)
				{
					ex->type = ex_free;
					break;
				}

				ent->alpha = (16.0f - (float)f) / 16.0f;

				if (f < 10)
				{
					ent->skinnum = (f >> 1);

					if (ent->skinnum < 0)
					{
						ent->skinnum = 0;
					}
				}
				else
				{
					ent->flags |= RF_TRANSLUCENT;

					if (f < 13)
					{
						ent->skinnum = 5;
					}

					else
					{
						ent->skinnum = 6;
					}
				}

				break;
			case ex_poly2:

				if (f >= ex->frames - 1)
				{
					ex->type = ex_free;
					break;
				}

				ent->alpha = (5.0 - (float)f) / 5.0;
				ent->skinnum = 0;
				ent->flags |= RF_TRANSLUCENT;
				break;
			default:
				break;
		}

		if (ex->type == ex_free)
		{
			continue;
		}

		if (ex->light)
		{
			V_AddLight(ent->origin, ex->light * ent->alpha,
					ex->lightcolor[0], ex->lightcolor[1], ex->lightcolor[2]);
		}

		VectorCopy(ent->origin, ent->oldorigin);

		if (f < 0)
		{
			f = 0;
		}

		ent->frame = ex->baseframe + f + 1;
		ent->oldframe = ex->baseframe + f;
		ent->backlerp = 1.0f - cl.lerpfrac;

		V_AddEntity(ent);
	}
}

void
CL_AddLasers(void)
{
	laser_t *l;
	int i;

	for (i = 0, l = cl_lasers; i < MAX_LASERS; i++, l++)
	{
		if (l->endtime >= cl.time)
		{
			V_AddEntity(&l->ent);
		}
	}
}

void
CL_ProcessSustain()
{
	cl_sustain_t *s;
	int i;

	for (i = 0, s = cl_sustains; i < MAX_SUSTAINS; i++, s++)
	{
		if (s->id)
		{
			if ((s->endtime >= cl.time) && (cl.time >= s->nextthink))
			{
				s->think(s);
			}
			else if (s->endtime < cl.time)
			{
				s->id = 0;
			}
		}
	}
}

void
CL_AddTEnts(void)
{
	CL_AddBeams();
	CL_AddPlayerBeams();
	CL_AddExplosions();
	CL_AddLasers();
	CL_ProcessSustain();
}

