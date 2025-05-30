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
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307,
 * USA.
 *
 * =======================================================================
 *
 * Header for the 2D client stuff and the .cinf file format
 *
 * =======================================================================
 */

#ifndef CL_SCREEN_H
#define CL_SCREEN_H

void	SCR_Init(void);

void	SCR_UpdateScreen(void);

void	SCR_SizeUp(void);
void	SCR_SizeDown(void);
void	SCR_CenterPrint(char *str);
void	SCR_BeginLoadingPlaque(void);
void	SCR_EndLoadingPlaque(void);

void	SCR_DebugGraph(float value, int color);

void	SCR_TouchPics(void);

void	SCR_RunConsole(void);

extern	int			sb_lines;

extern	cvar_t		*scr_viewsize;
extern	cvar_t		*crosshair;

extern	vrect_t		scr_vrect; /* position of render window */

void SCR_AddDirtyPoint(int x, int y);
void SCR_DirtyScreen(void);

void SCR_LoadImageWithPalette(const char *filename, byte **pic, byte **palette,
	int *width, int *height, int *bitsPerPixel);
void SCR_PlayCinematic(char *name);
qboolean SCR_DrawCinematic(void);
void SCR_RunCinematic(void);
void SCR_StopCinematic(void);
void SCR_FinishCinematic(void);

void SCR_DrawCrosshair(void);

float SCR_GetHUDScale(void);
float SCR_GetConsoleScale(void);
float SCR_GetMenuScale(void);

#endif
