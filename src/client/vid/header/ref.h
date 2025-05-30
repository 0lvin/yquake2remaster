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
 * ABI between client and refresher
 *
 * =======================================================================
 */

#ifndef CL_REF_H
#define CL_REF_H

#include "../../../common/header/common.h"
#include "vid.h"

#define	DLIGHT_CUTOFF	64
#define	MAX_DLIGHTS		32
#define	MAX_ENTITIES	128
#define	MAX_PARTICLES	4096
#define	MAX_LIGHTSTYLES	256

#define POWERSUIT_SCALE		4.0F

#define SHELL_RED_COLOR		0xF2
#define SHELL_GREEN_COLOR	0xD0
#define SHELL_BLUE_COLOR	0xF3

#define SHELL_RG_COLOR		0xDC
#define SHELL_RB_COLOR		0x68
#define SHELL_BG_COLOR		0x78

#define SHELL_DOUBLE_COLOR		0xDF
#define	SHELL_HALF_DAM_COLOR	0x90
#define SHELL_CYAN_COLOR		0x72

#define SHELL_WHITE_COLOR	0xD7

#define ENTITY_FLAGS	68

typedef struct entity_s {
	struct model_s		*model; /* opaque type outside refresh */
	float				angles[3];

	/* most recent data */
	float				origin[3]; /* also used as RF_BEAM's "from" */
	int					frame; /* also used as RF_BEAM's diameter */

	/* previous data for lerping */
	float				oldorigin[3]; /* also used as RF_BEAM's "to" */
	int					oldframe;

	/* misc */
	float	backlerp; /* 0.0 = current, 1.0 = old */
	int		skinnum; /* also used as RF_BEAM's palette index */
	vec3_t	scale; /* model scale before render */
	unsigned	rr_mesh; /* disabled meshes */

	int		lightstyle; /* for flashing entities */
	float	alpha; /* ignore if RF_TRANSLUCENT isn't set */

	struct image_s	*skin; /* NULL for inline skin */
	int		flags;
} entity_t;

typedef struct {
	vec3_t	origin;
	vec3_t	color;
	float	intensity;
} dlight_t;

typedef struct {
	vec3_t	origin;
	unsigned	color;
	float	alpha;
} particle_t;

typedef struct {
	float		rgb[3]; /* 0.0 - 2.0 */
	float		white; /* r+g+b */
} lightstyle_t;

typedef struct {
	int			x, y, width, height; /* in virtual screen coordinates */
	float		fov_x, fov_y;
	float		vieworg[3];
	float		viewangles[3];
	float		blend[4]; /* rgba 0-1 full screen blend */
	float		time; /* time is used to auto animate */
	int			rdflags; /* RDF_UNDERWATER, etc */

	byte		*areabits; /* if not NULL, only areas with set bits will be drawn */

	lightstyle_t	*lightstyles; /* [MAX_LIGHTSTYLES] */

	int			num_entities;
	entity_t	*entities;

	int			num_dlights; // <= 32 (MAX_DLIGHTS)
	dlight_t	*dlights;

	int			num_particles;
	particle_t	*particles;
} refdef_t;

// Renderer restart type.
typedef enum {
	RESTART_UNDEF,
	RESTART_NO,
	RESTART_FULL,
	RESTART_PARTIAL
} ref_restart_t;

// FIXME: bump API_VERSION?
#define	API_VERSION		7
#define EXPORT
#define IMPORT

//
// these are the functions exported by the refresh module
//
typedef struct
{
	// if api_version is different, the dll cannot be used
	int		api_version;

	// if framework_version is different, the dll cannot be used
	// necessary because differend SDL major version cannot be
	// mixed.
	int		framework_version;

	// called when the library is loaded
	qboolean (EXPORT *Init) (void);

	// called before the library is unloaded
	void	(EXPORT *Shutdown) (void);

	// called by GLimp_InitGraphics() before creating window,
	// returns flags for SDL window creation, returns -1 on error
	int		(EXPORT *PrepareForWindow)(void);

	// called by GLimp_InitGraphics() *after* creating window,
	// passing the SDL_Window* (void* so we don't spill SDL.h here)
	// (or SDL_Surface* for SDL1.2, another reason to use void*)
	// returns true (1) on success
	int		(EXPORT *InitContext)(void* sdl_window);

	// called by GLimp_InitGraphics() *after* creating render
	// context. Returns the actual drawable size in the width
	// and height variables. This may be different from the
	// window size due to high dpi awareness.
	void	(EXPORT *GetDrawableSize)(int* width, int* height);

	// shuts down rendering (OpenGL) context.
	void	(EXPORT *ShutdownContext)(void);

	// returns true if vsync is active, else false
	qboolean (EXPORT *IsVSyncActive)(void);

	// All data that will be used in a level should be
	// registered before rendering any frames to prevent disk hits,
	// but they can still be registered at a later time
	// if necessary.
	//
	// EndRegistration will free any remaining data that wasn't registered.
	// Any model_s or skin_s pointers from before the BeginRegistration
	// are no longer valid after EndRegistration.
	//
	// Skins and images need to be differentiated, because skins
	// are flood filled to eliminate mip map edge errors, and pics have
	// an implicit "pics/" prepended to the name. (a pic name that starts with a
	// slash will not use the "pics/" prefix or the ".pcx" postfix)
	void	(EXPORT *BeginRegistration) (const char *map);
	struct model_s * (EXPORT *RegisterModel) (const char *name);
	struct image_s * (EXPORT *RegisterSkin) (const char *name);

	void	(EXPORT *SetSky) (const char *name, float rotate, int autorotate, const vec3_t axis);
	void	(EXPORT *EndRegistration) (void);

	void	(EXPORT *RenderFrame) (refdef_t *fd);

	struct image_s * (EXPORT *DrawFindPic)(const char *name);

	void	(EXPORT *DrawGetPicSize) (int *w, int *h, const char *name);	// will return 0 0 if not found
	void 	(EXPORT *DrawPicScaled) (int x, int y, const char *pic, float factor, const char *alttext);
	void	(EXPORT *DrawStretchPic) (int x, int y, int w, int h, const char *name);
	void	(EXPORT *DrawCharScaled)(int x, int y, int num, float scale);
	void	(EXPORT *DrawStringScaled)(int x, int y, float scale, qboolean alt, const char *message);
	void	(EXPORT *DrawTileClear) (int x, int y, int w, int h, const char *name);
	void	(EXPORT *DrawFill) (int x, int y, int w, int h, int c);
	void	(EXPORT *DrawFadeScreen) (void);

	/*
	 * Draw images for cinematic rendering (which can have a different palette if bits equals to 8).
	 * Note that calls
	 */
	void	(EXPORT *DrawStretchRaw) (int x, int y, int w, int h, int cols, int rows, const byte *data, int bits);

	/*
	** video mode and refresh state management entry points
	*/
	void	(EXPORT *SetPalette)( const unsigned char *palette);	// NULL = game palette
	void	(EXPORT *BeginFrame)( float camera_separation );
	void	(EXPORT *EndFrame) (void);
	qboolean	(EXPORT *EndWorldRenderpass) (void); // finish world rendering, apply postprocess and switch to UI render pass

	//void	(EXPORT *AppActivate)( qboolean activate );
} refexport_t;

typedef struct
{
	YQ2_ATTR_NORETURN_FUNCPTR void	(IMPORT *Sys_Error) (int err_level, const char *str, ...) PRINTF_ATTR(2, 3);

	void	(IMPORT *Cmd_AddCommand) (const char *name, void(*cmd)(void));
	void	(IMPORT *Cmd_RemoveCommand) (const char *name);
	int		(IMPORT *Cmd_Argc) (void);
	char	*(IMPORT *Cmd_Argv) (int i);
	void	(IMPORT *Cmd_ExecuteText) (int exec_when, char *text);

	void	(IMPORT *Com_VPrintf) (int print_level, const char *fmt, va_list argptr);

	// files will be memory mapped read only
	// the returned buffer may be part of a larger pak file,
	// or a discrete file from anywhere in the quake search path
	// a -1 return means the file does not exist
	// NULL can be passed for buf to just determine existance
	int		(IMPORT *FS_LoadFile) (const char *name, void **buf);
	void	(IMPORT *FS_FreeFile) (void *buf);
	void	*(IMPORT *FS_AllocFile) (int size);

	// gamedir will be the current directory that generated
	// files should be stored to, ie: "f:\quake\id1"
	const char	*(IMPORT *FS_Gamedir) (void);

	cvar_t	*(IMPORT *Cvar_Get) (const char *name, const char *value, int flags);
	cvar_t	*(IMPORT *Cvar_Set) (const char *name, const char *value);
	void	 (IMPORT *Cvar_SetValue) (const char *name, float value);

	qboolean	(IMPORT *Vid_GetModeInfo)(int *width, int *height, int mode);
	void		(IMPORT *Vid_MenuInit)( void );
	// called with image data of width*height pixel which comp bytes per pixel (must be 3 or 4 for RGB or RGBA)
	// expects the pixels data to be row-wise, starting at top left
	void		(IMPORT *Vid_WriteScreenshot)( int width, int height, int comp, const void* data );
	/* load image from file */
	void		(IMPORT *VID_ImageDecode)( const char *filename, byte **pic, byte **palette,
				int *width, int *height, int *bitsPerPixel);
	void		(IMPORT *VID_GetPalette)(byte **colormap, unsigned *d_8to24table);
	void		(IMPORT *VID_GetPalette24to8)(const byte *d_8to24table, byte** d_16to8table);

	qboolean	(IMPORT *GLimp_InitGraphics)(int fullscreen, int *pwidth, int *pheight);
	qboolean	(IMPORT *GLimp_GetDesktopMode)(int *pwidth, int *pheight);

	void		(IMPORT *Vid_RequestRestart)(ref_restart_t rs);

	/* Rerelease: Get file from cache/converted */
	int (IMPORT *Mod_LoadFile)(const char *path, void **buffer);
} refimport_t;

// this is the only function actually exported at the linker level
typedef	refexport_t	(EXPORT *GetRefAPI_t) (refimport_t);

// FIXME: #ifdef client/ref around this
extern refexport_t re;
extern refimport_t ri;

/*
 * Refresh API
 */
void R_BeginRegistration(const char *map);
void R_Clear(void);
struct model_s *R_RegisterModel(const char *name);
struct image_s *R_RegisterSkin(const char *name);
void R_SetSky(const char *name, float rotate, int autorotate, const vec3_t axis);
void R_EndRegistration(void);
struct image_s *Draw_FindPic(const char *name);
void R_RenderFrame(refdef_t *fd);
void Draw_GetPicSize(int *w, int *h, const char *name);

void Draw_StretchPic(int x, int y, int w, int h, const char *name);
void Draw_PicScaled(int x, int y, const char *pic, float factor);
void Draw_PicScaledAltText(int x, int y, const char *pic, float factor, const char *alttext);

void Draw_CharScaled(int x, int y, int num, float scale);
void Draw_StringScaled(int x, int y, float scale, qboolean alt, const char *message);
void Draw_TileClear(int x, int y, int w, int h, const char *name);
void Draw_Fill(int x, int y, int w, int h, int c);
void Draw_FadeScreen(void);
void Draw_StretchRaw(int x, int y, int w, int h, int cols, int rows, const byte *data, int bits);
void R_SetPalette(const unsigned char *palette);
void R_BeginFrame(float camera_separation);
qboolean R_EndWorldRenderpass(void);
void R_EndFrame(void);

#endif
