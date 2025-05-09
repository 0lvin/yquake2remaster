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
 * Platform independent initialization, main loop and frame handling.
 *
 * =======================================================================
 */

#include "header/common.h"
#include "header/zone.h"
#include <setjmp.h>

cvar_t *developer;
cvar_t *modder;
cvar_t *timescale;
cvar_t *fixedtime;
cvar_t *cl_maxfps;
cvar_t *dedicated;

extern cvar_t *color_terminal;
extern cvar_t *logfile_active;
extern jmp_buf abortframe; /* an ERR_DROP occured, exit the entire frame */
extern zhead_t z_chain;

#ifndef DEDICATED_ONLY
FILE *log_stats_file;
cvar_t *busywait;
cvar_t *cl_async;
cvar_t *cl_timedemo;
cvar_t *vid_maxfps;
cvar_t *host_speeds;
cvar_t *log_stats;
cvar_t *showtrace;
#endif

// Forward declarations
#ifndef DEDICATED_ONLY
float GLimp_GetRefreshRate(void);
qboolean R_IsVSyncActive(void);
#endif

/* host_speeds times */
int time_before_game;
int time_after_game;
int time_before_ref;
int time_after_ref;

// Used in the network- and input paths.
int curtime;

#ifndef DEDICATED_ONLY
void Key_Init(void);
void Key_Shutdown(void);
void SCR_EndLoadingPlaque(void);
#endif

// Is the game portable?
qboolean is_portable;

// Game given by user
char userGivenGame[MAX_QPATH];

// Game should quit next frame.
// Hack for the signal handlers.
qboolean quitnextframe;

#ifndef DEDICATED_ONLY
#ifdef SDL_CPUPauseInstruction
#  define Sys_CpuPause() SDL_CPUPauseInstruction()
#else
static YQ2_ATTR_INLINE void Sys_CpuPause(void)
{
#if defined(__GNUC__)
#if (__i386 || __x86_64__)
	asm volatile("pause");
#elif defined(__aarch64__) || (defined(__ARM_ARCH) && __ARM_ARCH >= 7) || defined(__ARM_ARCH_6K__)
	asm volatile("yield");
#elif defined(__powerpc__) || defined(__powerpc64__)
	asm volatile("or 27,27,27");
#elif defined(__riscv) && __riscv_xlen == 64
	asm volatile(".insn i 0x0F, 0, x0, x0, 0x010");
#endif
#elif defined(_MSC_VER)
#if defined(_M_IX86) || defined(_M_X64)
	_mm_pause();
#elif defined(_M_ARM) || defined(_M_ARM64)
	__yield();
#endif
#endif
}
#endif
#endif

static void Qcommon_Frame(int usec);

// ----

static void
Qcommon_Buildstring(void)
{
	int i;
	size_t verLen;
	const char* versionString;


	versionString = va("Yamagi Quake II v%s", YQ2VERSION);
	verLen = strlen(versionString);

	printf("\n%s\n", versionString);

	for( i = 0; i < verLen; ++i)
	{
		printf("=");
	}

	printf("\n");


#ifndef DEDICATED_ONLY
	printf("Client build options:\n");

#ifdef USE_CURL
	printf(" + cURL HTTP downloads\n");
#else
	printf(" - cURL HTTP downloads\n");
#endif

#ifdef USE_OPENAL
	printf(" + OpenAL audio\n");
#else
	printf(" - OpenAL audio\n");
#endif

#ifdef SYSTEMWIDE
	printf(" + Systemwide installation\n");
#else
	printf(" - Systemwide installation\n");
#endif

#ifdef AVMEDIADECODE
	printf(" + AVcodec decode\n");
#else
	printf(" - AVcodec decode\n");
#endif
#endif

	printf("Platform: %s\n", YQ2OSTYPE);
	printf("Architecture: %s\n", YQ2ARCH);
}

static void
Qcommon_Mainloop(void)
{
	long long newtime;
	long long oldtime = Sys_Microseconds();

	/* The mainloop. The legend. */
	while (1)
	{
#ifndef DEDICATED_ONLY
		if (!cl_timedemo->value)
		{
			// Throttle the game a little bit.
			if (busywait->value)
			{
				long long spintime = Sys_Microseconds();

				while (1)
				{
					/* Give the CPU a hint that this is a very tight
					   spinloop. One PAUSE instruction each loop is
					   enough to reduce power consumption and head
					   dispersion a lot, it's 95°C against 67°C on
					   a Kaby Lake laptop. */
					Sys_CpuPause();

					if (Sys_Microseconds() - spintime >= 5)
					{
						break;
					}
				}
			}
			else
			{
				Sys_Nanosleep(5000);
			}
		}
#else
		Sys_Nanosleep(850000);
#endif

		newtime = Sys_Microseconds();

		// Save global time for network- und input code.
		curtime = (int)(newtime / 1000ll);

		Qcommon_Frame(newtime - oldtime);
		oldtime = newtime;
	}
}

void Qcommon_ExecConfigs(qboolean gameStartUp)
{
	Cbuf_AddText("exec default.cfg\n");
	Cbuf_AddText("exec yq2.cfg\n");
	Cbuf_AddText("exec config.cfg\n");
	Cbuf_AddText("exec autoexec.cfg\n");

	if (gameStartUp)
	{
		/* Process cmd arguments only startup. */
		Cbuf_AddEarlyCommands(true);
	}

	Cbuf_Execute();
}

static qboolean checkForHelp(int argc, char **argv)
{
	const char* helpArgs[] = { "--help", "-h", "-help", "-?", "/?" };
	const int numHelpArgs = sizeof(helpArgs)/sizeof(helpArgs[0]);

	for (int i=1; i<argc; ++i)
	{
		const char* arg = argv[i];

		for (int h=0; h<numHelpArgs; ++h)
		{
			if (Q_stricmp(arg, helpArgs[h]) == 0)
			{
				printf("Yamagi Quake II v%s\n", YQ2VERSION);
				printf("Most interesting commandline arguments:\n");
				printf("-h or --help: Show this help\n");
				printf("-cfgdir <path>\n");
				printf("  set the name of your config directory\n");
				printf("-datadir <path>\n");
				printf("  set path to your Quake2 game data (the directory baseq2/ is in)\n");
				printf("-portable\n");
				printf("  Write (savegames, configs, ...) in the binary directory\n");
				printf("+exec <config>\n");
				printf("  execute the given config (mainly relevant for dedicated servers)\n");
				printf("+set <cvarname> <value>\n");
				printf("  Set the given cvar to the given value, e.g. +set vid_fullscreen 0\n");

				printf("\nSome interesting cvars:\n");
				printf("+set game <gamename>\n");
				printf("  start the given addon/mod, e.g. +set game xatrix\n");
#ifndef DEDICATED_ONLY
				printf("+set vid_fullscreen <0 or 1>\n");
				printf("  start game in windowed (0) or desktop fullscreen (1)\n");
				printf("  or classic fullscreen (2) mode\n");
				printf("+set r_mode <modenumber>\n");
				printf("  start game in resolution belonging to <modenumber>,\n");
				printf("  use -1 for custom resolutions:\n");
				printf("+set r_customwidth <size in pixels>\n");
				printf("+set r_customheight <size in pixels>\n");
				printf("  if r_mode is set to -1, these cvars allow you to specify the\n");
				printf("  width/height of your custom resolution\n");
				printf("+set vid_renderer <renderer>\n");
				printf("  Selects the render backend. Currently available:\n");
				printf("    'gl1'  (the OpenGL 1.x renderer),\n");
				printf("    'gl3'  (the OpenGL 3.2 renderer),\n");
				printf("    'soft' (the software renderer)\n");
#endif // DEDICATED_ONLY
				printf("\nSee https://github.com/yquake2/yquake2/blob/master/doc/04_cvarlist.md\nfor some more cvars\n");

				return true;
			}
		}
	}

	return false;
}

void
Qcommon_Init(int argc, char **argv)
{
	// Jump point used in emergency situations.
	if (setjmp(abortframe))
	{
		Sys_Error("Error during initialization");
	}

	if (checkForHelp(argc, argv))
	{
		// ok, --help or similar commandline option was given
		// and info was printed, exit the game now
		exit(1);
	}

	// Print the build and version string
	Qcommon_Buildstring();

	// Seed PRNG
	randk_seed();

	// Initialize zone malloc().
	z_chain.next = z_chain.prev = &z_chain;

	// Start early subsystems.
	COM_InitArgv(argc, argv);
	Swap_Init();
	Cbuf_Init();
	Cmd_Init();
	Cvar_Init();

#ifndef DEDICATED_ONLY
	Key_Init();
#endif

	/* we need to add the early commands twice, because
	   a basedir or cddir needs to be set before execing
	   config files, but we want other parms to override
	   the settings of the config files */
	Cbuf_AddEarlyCommands(false);
	Cbuf_Execute();
	/* Set default maptype  */
	Cvar_Get("maptype", "0", CVAR_ARCHIVE);

	// remember the initial game name that might have been set on commandline
	{
		cvar_t* gameCvar, *gametypeCvar;
		const char* game = "";

		gameCvar = Cvar_Get("game", "", CVAR_LATCH | CVAR_SERVERINFO);
		gametypeCvar = Cvar_Get("gametype", "", CVAR_LATCH | CVAR_SERVERINFO);

		if(gameCvar->string && gameCvar->string[0])
		{
			game = gameCvar->string;
			if (strcmp(gametypeCvar->string, gameCvar->string))
			{
				/* Set gametype if game is provided */
				Cvar_Set("gametype", gameCvar->string);
			}
		}

		Q_strlcpy(userGivenGame, game, sizeof(userGivenGame));
	}

	// The filesystems needs to be initialized after the cvars.
	FS_InitFilesystem();
	Mod_AliasesInit();
	CM_ModInit();

	// Add and execute configuration files.
	Qcommon_ExecConfigs(true);

	// Zone malloc statistics.
	Cmd_AddCommand("z_stats", Z_Stats_f);

	// cvars

	cl_maxfps = Cvar_Get("cl_maxfps", "-1", CVAR_ARCHIVE);

	developer = Cvar_Get("developer", "0", 0);
	fixedtime = Cvar_Get("fixedtime", "0", 0);

	color_terminal = Cvar_Get("colorterminal", "1", CVAR_ARCHIVE);
	logfile_active = Cvar_Get("logfile", "1", CVAR_ARCHIVE);
	modder = Cvar_Get("modder", "0", 0);
	timescale = Cvar_Get("timescale", "1", 0);

	char *s;
	s = va("%s %s %s %s", YQ2VERSION, YQ2ARCH, BUILD_DATE, YQ2OSTYPE);
	Cvar_Get("version", s, CVAR_SERVERINFO | CVAR_NOSET);

#ifndef DEDICATED_ONLY
	busywait = Cvar_Get("busywait", "1", CVAR_ARCHIVE);
	cl_async = Cvar_Get("cl_async", "1", CVAR_ARCHIVE);
	cl_timedemo = Cvar_Get("timedemo", "0", 0);
	dedicated = Cvar_Get("dedicated", "0", CVAR_NOSET);
	vid_maxfps = Cvar_Get("vid_maxfps", "300", CVAR_ARCHIVE);
	host_speeds = Cvar_Get("host_speeds", "0", 0);
	log_stats = Cvar_Get("log_stats", "0", 0);
	showtrace = Cvar_Get("showtrace", "0", 0);
#else
	dedicated = Cvar_Get("dedicated", "1", CVAR_NOSET);
#endif

	// We can't use the clients "quit" command when running dedicated.
	if (dedicated->value)
	{
		Cmd_AddCommand("quit", Com_Quit);
	}

	// Start late subsystem.
	Sys_Init();
	NET_Init();
	Netchan_Init();
	SV_Init();
#ifndef DEDICATED_ONLY
	CL_Init();
#endif

	// Everythings up, let's add + cmds from command line.
	if (!Cbuf_AddLateCommands())
	{
		if (!dedicated->value)
		{
			// Start demo loop...
			Cbuf_AddText("d1\n");
		}
		else
		{
			// ...or dedicated server.
			Cbuf_AddText("dedicated_start\n");
		}

		Cbuf_Execute();
	}
#ifndef DEDICATED_ONLY
	else
	{
		/* the user asked for something explicit
		   so drop the loading plaque */
		SCR_EndLoadingPlaque();
	}
#endif

	Com_Printf("==== Yamagi Quake II Initialized ====\n\n");
	Com_Printf("*************************************\n\n");

	// Call the main loop
	Qcommon_Mainloop();
}

#ifndef DEDICATED_ONLY
static void
Qcommon_Frame(int usec)
{
	// Used for the dedicated server console.
	char *s;

	// Statistics.
	int time_before = 0;
	int time_between = 0;
	int time_after;

	// Target packetframerate.
	float pfps;

	// Target renderframerate.
	float rfps;

	// Time since last packetframe in microsec.
	static int packetdelta = 1000000;

	// Time since last renderframe in microsec.
	static int renderdelta = 1000000;

	// Accumulated time since last client run.
	static int clienttimedelta = 0;

	// Accumulated time since last server run.
	static int servertimedelta = 0;

	/* A packetframe runs the server and the client,
	   but not the renderer. The minimal interval of
	   packetframes is about 10.000 microsec. If run
	   more often the movement prediction in pmove.c
	   breaks. That's the Q2 variant if the famous
	   125hz bug. */
	qboolean packetframe = true;

	/* A rendererframe runs the renderer, but not the
	   client or the server. The minimal interval is
	   about 1000 microseconds. */
	qboolean renderframe = true;


	/* Tells the client to shutdown.
	   Used by the signal handlers. */
	if (quitnextframe)
	{
		Cbuf_AddText("quit");
	}


	/* In case of ERR_DROP we're jumping here. Don't know
	   if that's really save but it seems to work. So leave
	   it alone. */
	if (setjmp(abortframe))
	{
		return;
	}


	if (log_stats->modified)
	{
		log_stats->modified = false;

		if (log_stats->value)
		{
			if (log_stats_file)
			{
				fclose(log_stats_file);
				log_stats_file = 0;
			}

			log_stats_file = Q_fopen("stats.log", "w");

			if (log_stats_file)
			{
				fprintf(log_stats_file, "entities,dlights,parts,frame time\n");
			}
		}
		else
		{
			if (log_stats_file)
			{
				fclose(log_stats_file);
				log_stats_file = 0;
			}
		}
	}


	// Timing debug crap. Just for historical reasons.
	if (fixedtime->value)
	{
		usec = (int)fixedtime->value;
	}
	else if (timescale->value)
	{
		usec *= timescale->value;
	}


	if (showtrace->value)
	{
		extern int c_traces, c_brush_traces;
		extern int c_pointcontents;

		Com_Printf("%4i traces  %4i points\n", c_traces, c_pointcontents);
		c_traces = 0;
		c_brush_traces = 0;
		c_pointcontents = 0;
	}


	/* We can render 1000 frames at maximum, because the minimum
	   frametime of the client is 1 millisecond. And of course we
	   need to render something, the framerate can never be less
	   then 1. Cap vid_maxfps between 1 and 999. */
	if (vid_maxfps->value > 999 || vid_maxfps->value < 1)
	{
		Cvar_SetValue("vid_maxfps", 999);
	}

	if (cl_maxfps->value > 250)
	{
		Cvar_SetValue("cl_maxfps", 250);
	}

	// Calculate target and renderframerate.
	if (R_IsVSyncActive())
	{
		float refreshrate = GLimp_GetRefreshRate();

		// using refreshRate - 2, because targeting a value slightly below the
		// (possibly not 100% correctly reported) refreshRate would introduce jittering, so only
		// use vid_maxfps if it looks like the user really means it to be different from refreshRate
		if (vid_maxfps->value < refreshrate - 2 )
		{
			rfps = vid_maxfps->value;
			// we can't have more packet frames than render frames, so limit pfps to rfps
			pfps = (cl_maxfps->value > rfps) ? rfps : cl_maxfps->value;
		}
		else // target refresh rate, not vid_maxfps
		{
			/* if vsync is active, we increase the target framerate a bit for two reasons
			   1. On Windows, GLimp_GetFrefreshRate() (or the SDL counterpart, or even
			      the underlying WinAPI function) often returns a too small value,
			      like 58 or 59 when it's really 59.95 and thus (as integer) should be 60
			   2. vsync will throttle us to refreshrate anyway, so there is no harm
			      in starting the frame *a bit* earlier, instead of risking starting
			      it too late */
			rfps = refreshrate * 1.2f;
			// we can't have more packet frames than render frames, so limit pfps to rfps
			// but in this case use tolerance for comparison and assign rfps with tolerance
			pfps = (cl_maxfps->value < refreshrate - 2) ? cl_maxfps->value : rfps;
		}
	}
	else
	{
		rfps = vid_maxfps->value;
		// we can't have more packet frames than render frames, so limit pfps to rfps
		pfps = (cl_maxfps->value > rfps) ? rfps : cl_maxfps->value;
	}

	// cl_maxfps <= 0 means: automatically choose a packet framerate that should work
	// well with the render framerate, which is the case if rfps is a multiple of pfps
	if (cl_maxfps->value <= 0.0f && cl_async->value != 0.0f)
	{
		// packet framerates between about 45 and 90 should be ok,
		// with other values the game (esp. movement/clipping) can become glitchy
		// as pfps must be <= rfps, for rfps < 90 just use that as pfps
		if (rfps < 90.0f)
		{
			pfps = rfps;
		}
		else
		{
			/* we want an integer divider, so every div-th renderframe is a packetframe.
			   this formula gives nice dividers that keep pfps as close as possible
			   to 60 (which seems to be ideal):
			   - for < 150 rfps div will be 2, so pfps will be between 45 and ~75
			     => exactly every second renderframe we also run a packetframe
			   - for < 210 rfps div will be 3, so pfps will be between 50 and ~70
			     => exactly every third renderframe we also run a packetframe
			   - etc, the higher the rfps, the closer the pfps-range will be to 60
			     (and you probably get the very best results by running at a
			      render framerate that's a multiple of 60) */
			float div = round(rfps/60);
			pfps = rfps/div;
		}
	}

	// Calculate timings.
	packetdelta += usec;
	renderdelta += usec;
	clienttimedelta += usec;
	servertimedelta += usec;

	if (!cl_timedemo->value)
	{
		if (cl_async->value)
		{
			// Render frames.
			if (renderdelta < (1000000.0f / rfps))
			{
				renderframe = false;
			}

			// Network frames.
			float packettargetdelta = 1000000.0f / pfps;
			// "packetdelta + renderdelta/2 >= packettargetdelta" if now we're
			// closer to when we want to run the next packetframe than we'd
			// (probably) be after the next render frame
			// also, we only run packetframes together with renderframes,
			// because we must have at least one render frame between two packet frames
			// TODO: does it make sense to use the average renderdelta of the last X frames
			//       instead of just the last renderdelta?
			if (!renderframe || packetdelta + renderdelta/2 < packettargetdelta)
			{
				packetframe = false;
			}
		}
		else
		{
			// Cap frames at target framerate.
			if (renderdelta < (1000000.0f / rfps)) {
				renderframe = false;
				packetframe = false;
			}
		}
	}

	// Dedicated server terminal console.
	do {
		s = Sys_ConsoleInput();

		if (s) {
			Cbuf_AddText(va("%s\n", s));
		}
	} while (s);

	Cbuf_Execute();


	if (host_speeds->value)
	{
		time_before = Sys_Milliseconds();
	}


	// Run the serverframe.
	if (packetframe) {
		SV_Frame(servertimedelta);
		servertimedelta = 0;
	}


	if (host_speeds->value)
	{
		time_between = Sys_Milliseconds();
	}


	// Run the client frame.
	if (packetframe || renderframe) {
		CL_Frame(packetdelta, renderdelta, clienttimedelta, packetframe, renderframe);
		clienttimedelta = 0;
	}


	if (host_speeds->value)
	{
		int all, sv, gm, cl, rf;

		time_after = Sys_Milliseconds();
		all = time_after - time_before;
		sv = time_between - time_before;
		cl = time_after - time_between;
		gm = time_after_game - time_before_game;
		rf = time_after_ref - time_before_ref;
		sv -= gm;
		cl -= rf;
		Com_Printf("all:%3i sv:%3i gm:%3i cl:%3i rf:%3i\n", all, sv, gm, cl, rf);
	}


	// Reset deltas and mark frame.
	if (packetframe) {
		packetdelta = 0;
	}

	if (renderframe) {
		renderdelta = 0;
	}
}
#else
static void
Qcommon_Frame(int usec)
{
	// For the dedicated server terminal console.
	char *s;

	// Target packetframerate.
	int pfps;

	// Time since last packetframe in microsec.
	static int packetdelta = 1000000;

	// Accumulated time since last server run.
	static int servertimedelta = 0;

	/* A packetframe runs the server and the client,
	   but not the renderer. The minimal interval of
	   packetframes is about 10.000 microsec. If run
	   more often the movement prediction in pmove.c
	   breaks. That's the Q2 variant if the famous
	   125hz bug. */
	qboolean packetframe = true;


	/* Tells the client to shutdown.
	   Used by the signal handlers. */
	if (quitnextframe)
	{
		Cbuf_AddText("quit");
	}


	/* In case of ERR_DROP we're jumping here. Don't know
	   if that' really save but it seems to work. So leave
	   it alone. */
	if (setjmp(abortframe))
	{
		return;
	}


	// Timing debug crap. Just for historical reasons.
	if (fixedtime->value)
	{
		usec = (int)fixedtime->value;
	}
	else if (timescale->value)
	{
		usec *= timescale->value;
	}


	// Target framerate.
	pfps = (int)cl_maxfps->value;


	// Calculate timings.
	packetdelta += usec;
	servertimedelta += usec;


	// Network frame time.
	if (packetdelta < (1000000.0f / pfps)) {
		packetframe = false;
	}


	// Dedicated server terminal console.
	do {
		s = Sys_ConsoleInput();

		if (s) {
			Cbuf_AddText(va("%s\n", s));
		}
	} while (s);

	Cbuf_Execute();


	// Run the serverframe.
	if (packetframe) {
		SV_Frame(servertimedelta);
		servertimedelta = 0;

		// Reset deltas if necessary.
		packetdelta = 0;
	}
}
#endif

void
Qcommon_Shutdown(void)
{
	CM_ModFreeAll();
	Mod_AliasesFreeAll();
	FS_ShutdownFilesystem();
	Cvar_Fini();

#ifndef DEDICATED_ONLY
	Key_Shutdown();
#endif

	Cmd_Shutdown();
}
