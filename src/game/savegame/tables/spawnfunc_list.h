/*
 * Copyright (C) 1997-2001 Id Software, Inc.
 * Copyright (C) 2011 Yamagi Burmeister
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
 * =======================================================================
 *
 * Functionpointers to every spawn function in the game.so.
 *
 * =======================================================================
 */

{"ammo_magslug", SP_xatrix_item},
{"ammo_trap", SP_xatrix_item},
{"choose_cdtrack", SP_choose_cdtrack},
{"dm_dball_ball", SP_dm_dball_ball},
{"dm_dball_ball_start", SP_dm_dball_ball_start},
{"dm_dball_goal", SP_dm_dball_goal},
{"dm_dball_speed_change", SP_dm_dball_speed_change},
{"dm_dball_team1_start", SP_dm_dball_team1_start},
{"dm_dball_team2_start", SP_dm_dball_team2_start},
{"dm_tag_token", SP_dm_tag_token},
{"env_fire", SP_env_fire},
{"func_areaportal", SP_func_areaportal},
{"func_button", SP_func_button},
{"func_clock", SP_func_clock},
{"func_conveyor", SP_func_conveyor},
{"func_door", SP_func_door},
{"func_door_rotating", SP_func_door_rotating},
{"func_door_secret", SP_func_door_secret},
{"func_door_secret2", SP_func_door_secret2},
{"func_explosive", SP_func_explosive},
{"func_force_wall", SP_func_force_wall},
{"func_group", SP_info_null},
{"func_killbox", SP_func_killbox},
{"func_object", SP_func_object},
{"func_object_repair", SP_object_repair},
{"func_plat", SP_func_plat},
{"func_plat2", SP_func_plat2},
{"func_rotating", SP_func_rotating},
{"func_timer", SP_func_timer},
{"func_train", SP_func_train},
{"func_wall", SP_func_wall},
{"func_water", SP_func_water},
{"hint_path", SP_hint_path},
{"info_notnull", SP_info_notnull},
{"info_null", SP_info_null},
{"info_player_coop", SP_info_player_coop},
{"info_player_coop_lava", SP_info_player_coop_lava},
{"info_player_deathmatch", SP_info_player_deathmatch},
{"info_player_intermission", SP_info_player_intermission},
{"info_player_start", SP_info_player_start},
{"info_player_team1", SP_info_player_team1},
{"info_player_team2", SP_info_player_team2},
{"info_teleport_destination", SP_info_teleport_destination},
{"item_health", SP_item_health},
{"item_health_large", SP_item_health_large},
{"item_health_mega", SP_item_health_mega},
{"item_health_small", SP_item_health_small},
{"item_quadfire", SP_xatrix_item},
{"light", SP_light},
{"light_mine1", SP_light_mine1},
{"light_mine2", SP_light_mine2},
{"misc_actor", SP_misc_actor},
{"misc_amb4", SP_misc_amb4},
{"misc_banner", SP_misc_banner},
{"misc_bigviper", SP_misc_bigviper},
{"misc_blackhole", SP_misc_blackhole},
{"misc_crashviper", SP_misc_crashviper},
{"misc_ctf_banner", SP_misc_ctf_banner},
{"misc_ctf_small_banner", SP_misc_ctf_small_banner},
{"misc_deadsoldier", SP_misc_deadsoldier},
{"misc_easterchick", SP_misc_easterchick},
{"misc_easterchick2", SP_misc_easterchick2},
{"misc_eastertank", SP_misc_eastertank},
{"misc_explobox", SP_misc_explobox},
{"misc_gib_arm", SP_misc_gib_arm},
{"misc_gib_head", SP_misc_gib_head},
{"misc_gib_leg", SP_misc_gib_leg},
{"misc_insane", SP_misc_insane},
{"misc_nuke", SP_misc_nuke},
{"misc_nuke_core", SP_misc_nuke_core},
{"misc_flare", SP_misc_flare},
{"misc_model", SP_misc_model},
{"misc_satellite_dish", SP_misc_satellite_dish},
{"misc_strogg_ship", SP_misc_strogg_ship},
{"misc_teleporter", SP_misc_teleporter},
{"misc_teleporter_dest", SP_misc_teleporter_dest},
{"misc_transport", SP_misc_transport},
{"misc_viper", SP_misc_viper},
{"misc_viper_bomb", SP_misc_viper_bomb},
{"misc_viper_missile", SP_misc_viper_missile},
{"monster_arachnid", SP_monster_arachnid},
{"monster_army", SP_monster_army},
{"monster_berserk", SP_monster_berserk},
{"monster_boss2", SP_monster_boss2},
{"monster_boss3_stand", SP_monster_boss3_stand},
{"monster_boss5", SP_monster_boss5},
{"monster_brain", SP_monster_brain},
{"monster_carrier", SP_monster_carrier},
{"monster_chick", SP_monster_chick},
{"monster_chick_heat", SP_monster_chick_heat},
{"monster_commander_body", SP_monster_commander_body},
{"monster_daedalus", SP_monster_hover},
{"monster_demon", SP_monster_demon},
{"monster_dog", SP_monster_dog},
{"monster_enforcer", SP_monster_enforcer},
{"monster_rotfish", SP_monster_rotfish},
{"monster_fixbot", SP_monster_fixbot},
{"monster_flipper", SP_monster_flipper},
{"monster_floater", SP_monster_floater},
{"monster_flyer", SP_monster_flyer},
{"monster_gekk", SP_monster_gekk},
{"monster_gladb", SP_monster_gladb},
{"monster_gladiator", SP_monster_gladiator},
{"monster_guardian", SP_monster_guardian},
{"monster_guncmdr", SP_monster_guncmdr},
{"monster_gunner", SP_monster_gunner},
{"monster_hknight", SP_monster_hknight},
{"monster_hover", SP_monster_hover},
{"monster_infantry", SP_monster_infantry},
{"monster_jorg", SP_monster_jorg},
{"monster_kamikaze", SP_monster_kamikaze},
{"monster_knight", SP_monster_knight},
{"monster_makron", SP_monster_makron},
{"monster_medic", SP_monster_medic},
{"monster_medic_commander", SP_monster_medic},
{"monster_mutant", SP_monster_mutant},
{"monster_ogre", SP_monster_ogre},
{"monster_parasite", SP_monster_parasite},
{"monster_shalrath", SP_monster_shalrath},
{"monster_shambler", SP_monster_shambler},
{"monster_soldier", SP_monster_soldier},
{"monster_soldier_hypergun", SP_monster_soldier_hypergun},
{"monster_soldier_lasergun", SP_monster_soldier_lasergun},
{"monster_soldier_light", SP_monster_soldier_light},
{"monster_soldier_ripper", SP_monster_soldier_ripper},
{"monster_soldier_ss", SP_monster_soldier_ss},
{"monster_stalker", SP_monster_stalker},
{"monster_supertank", SP_monster_supertank},
{"monster_tank", SP_monster_tank},
{"monster_tank_commander", SP_monster_tank},
{"monster_tank_stand", SP_monster_tank},
{"monster_tarbaby", SP_monster_tarbaby},
{"monster_turret", SP_monster_turret},
{"monster_widow", SP_monster_widow},
{"monster_widow2", SP_monster_widow2},
{"monster_wizard", SP_monster_wizard},
{"monster_zombie", SP_monster_zombie},
{"npc_timeminder", SP_npc_timeminder},
{"object_big_fire", SP_object_big_fire},
{"object_campfire", SP_object_campfire},
{"object_flame1", SP_object_flame1},
{"path_corner", SP_path_corner},
{"point_combat", SP_point_combat},
{"rotating_light", SP_rotating_light},
{"target_actor", SP_target_actor},
{"target_anger", SP_target_anger},
{"target_autosave", SP_target_autosave},
{"target_blacklight", SP_target_blacklight},
{"target_blaster", SP_target_blaster},
{"target_camera", SP_target_camera},
{"target_changelevel", SP_target_changelevel},
{"target_character", SP_target_character},
{"target_crosslevel_target", SP_target_crosslevel_target},
{"target_crosslevel_trigger", SP_target_crosslevel_trigger},
{"target_earthquake", SP_target_earthquake},
{"target_explosion", SP_target_explosion},
{"trigger_flashlight", SP_trigger_flashlight},
{"target_goal", SP_target_goal},
{"target_gravity", SP_target_gravity},
{"target_help", SP_target_help},
{"target_killplayers", SP_target_killplayers},
{"target_laser", SP_target_laser},
{"target_lightramp", SP_target_lightramp},
{"target_mal_laser", SP_target_mal_laser},
{"target_music", SP_target_music},
{"target_orb", SP_target_orb},
{"target_soundfx", SP_target_soundfx},
{"target_secret", SP_target_secret},
{"target_sky", SP_target_sky},
{"target_spawner", SP_target_spawner},
{"target_speaker", SP_target_speaker},
{"target_splash", SP_target_splash},
{"target_steam", SP_target_steam},
{"target_string", SP_target_string},
{"target_temp_entity", SP_target_temp_entity},
{"trigger_always", SP_trigger_always},
{"trigger_counter", SP_trigger_counter},
{"trigger_disguise", SP_trigger_disguise},
{"trigger_elevator", SP_trigger_elevator},
{"trigger_gravity", SP_trigger_gravity},
{"trigger_hurt", SP_trigger_hurt},
{"trigger_key", SP_trigger_key},
{"trigger_monsterjump", SP_trigger_monsterjump},
{"trigger_multiple", SP_trigger_multiple},
{"trigger_once", SP_trigger_once},
{"trigger_push", SP_trigger_push},
{"trigger_relay", SP_trigger_relay},
{"trigger_teleport", SP_trigger_teleport},
{"turret_base", SP_turret_base},
{"turret_breach", SP_turret_breach},
{"turret_driver", SP_turret_driver},
{"turret_invisible_brain", SP_turret_invisible_brain},
{"viewthing", SP_viewthing},
{"weapon_boomer", SP_xatrix_item},
{"weapon_phalanx", SP_xatrix_item},
{"worldspawn", SP_worldspawn},
{NULL, NULL}
