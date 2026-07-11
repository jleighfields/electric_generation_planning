# set up logging
import logging
import os
import shutil
import sqlite3

import pandas as pd

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Name of the zip archive of results written for the download button.
RESULTS_ZIP = "results.zip"


class ResultsDB:
    """Per-session in-memory SQLite store for solved run results.

    Each saved run is spread across four tables keyed by run name
    (``table_names``); the store is created fresh per Shiny session, so
    nothing persists to disk between sessions.
    """

    def __init__(self) -> None:
        """Open the in-memory database and record the results-schema tables."""
        # check_same_thread=False: the background solve thread and the UI
        # thread both touch this connection within one session.
        self.db = sqlite3.connect(":memory:", check_same_thread=False)
        # Canonical results-schema table names (single source of truth for
        # the keys the model, store, and UI share).
        self.table_names = ["inputs", "cap_mw", "metrics", "final_df"]

    def clear_db(self) -> None:
        """Delete all rows from every results table that exists."""
        cur = None
        try:
            cur = self.db.cursor()
            sql_text = "SELECT name FROM sqlite_master WHERE type='table';"
            tables = cur.execute(sql_text).fetchall()
            tables = [t[0] for t in tables]
            for table_name in self.table_names:
                if table_name in tables:
                    sql_text = f"DELETE from {table_name}"
                    log.info(f"sql_text: {sql_text}")
                    cur.execute(sql_text)

        except Exception as e:
            log.error("Error thrown during clear_db")
            log.error(e)

        finally:
            if cur:
                cur.close()

    def add_run(self, results: dict) -> None:
        """Persist one run's results across the four results tables.

        A run with a name that already exists is deleted first so the save
        overwrites rather than duplicates. The scalar tables (``inputs``,
        ``cap_mw``, ``metrics``) are stored as a single row indexed by run
        name; ``final_df`` is stored with the run name added as its index.

        Args:
            results: The dict returned by LP.run_lp(), with keys
                ``run_name``, ``inputs``, ``obj_val``, ``cap_mw``,
                ``metrics``, and ``final_df``.
        """
        try:
            if results["run_name"] in self.get_runs():
                log.info(f"deleting old run {results['run_name']}")
                self.delete_run(results["run_name"])

            for table_name in self.table_names:
                log.info(f"adding {table_name} for run {results['run_name']}")
                if table_name != "final_df":
                    df = pd.DataFrame(results[table_name], index=[results["run_name"]])
                    df.index.names = ["run_name"]
                else:
                    df = results[table_name].copy()
                    df["run_name"] = results["run_name"]
                    df.set_index("run_name", inplace=True)
                df.to_sql(name=table_name, con=self.db, if_exists="append", index=True)

        except Exception as e:
            log.error("Error thrown during add run data")
            log.error(e)

    def delete_run(self, run: str) -> None:
        """Delete every row for a run name from all results tables.

        Args:
            run: The run name to remove.
        """
        cur = None
        try:
            cur = self.db.cursor()
            for table_name in self.table_names:
                sql_text = f"DELETE from {table_name} WHERE run_name = ?"
                cur.execute(sql_text, (run,))

        except Exception as e:
            log.error("Error thrown during delete_run")
            log.error(e)

        finally:
            if cur:
                cur.close()

    def get_runs(self):
        """Return the unique saved run names.

        Returns:
            A numpy array of run names from the ``inputs`` table, or an
            empty list if no runs have been saved yet.
        """
        cur = self.db.cursor()
        sql_text = "SELECT name FROM sqlite_master WHERE type='table';"
        tables = cur.execute(sql_text).fetchall()
        tables = [t[0] for t in tables]
        if "inputs" in tables:
            sql_text = "select run_name from inputs"
            temp_df = pd.read_sql_query(sql_text, self.db)
            res = temp_df.run_name.unique()
        else:
            res = []

        if cur:
            cur.close()

        return res

    def zip_results(self) -> None:
        """Write each results table to a CSV under ./csv and archive them.

        Produces ``results.zip`` in the working directory (ephemeral
        storage) for the download button. Overwrites any prior ./csv
        contents on each call.
        """
        log.info("zipping results")
        zip_loc = "./csv"
        if os.path.exists(zip_loc):
            shutil.rmtree(zip_loc)
        os.mkdir(zip_loc)

        cur = None
        try:
            cur = self.db.cursor()
            sql_text = "SELECT name FROM sqlite_master WHERE type='table';"
            tables = cur.execute(sql_text).fetchall()
            tables = [t[0] for t in tables]
            for table_name in self.table_names:
                if table_name in tables:
                    sql_text = f"SELECT * from {table_name}"
                    save_path = f"{zip_loc}/{table_name}.csv"
                    df = pd.read_sql_query(sql_text, self.db)
                    df.to_csv(save_path)
                    log.info(f"zipping table: {table_name}")
                    log.info(f"n_rows: {df.shape[0]}")

            shutil.make_archive(RESULTS_ZIP.removesuffix(".zip"), "zip", zip_loc)

        except Exception as e:
            log.error("Error thrown during zipping results")
            log.error(e)
        finally:
            if cur:
                cur.close()

    def print_head(self) -> None:
        """Log the first rows of each results table (debug helper)."""
        for table_name in self.table_names:
            sql_text = f"SELECT * from {table_name}"
            temp_df = pd.read_sql_query(sql_text, self.db)
            log.info(f"table: {table_name}")
            log.info(temp_df.head())
            print("\n")


def make_fake_results(run_name: str) -> dict:
    """Build a minimal results dict (same shape as LP.run_lp) for testing.

    Args:
        run_name: The run name to embed in the fake results.

    Returns:
        A results dict with the ``run_name``, ``inputs``, ``obj_val``,
        ``cap_mw``, ``metrics``, and ``final_df`` keys run_lp returns.
    """
    return {
        "run_name": run_name,
        "inputs": {"peak_load": 1000, "min_obj": "minimize cost"},
        "obj_val": 1.0,
        "cap_mw": {"batt_mw": 100, "solar_mw": 200, "wind_mw": 300, "gas_mw": 50},
        "metrics": {"total_cost_mill": 10.0, "gas_percent": 5.0},
        "final_df": pd.DataFrame({"2030_load": [1.0, 2.0], "gas": [0.0, 0.0]}),
    }


if __name__ == "__main__":
    print("\n")

    log.info("ADD test runs")
    test_db = ResultsDB()
    test_db.clear_db()
    test_db.add_run(make_fake_results("test"))
    test_db.add_run(make_fake_results("test2"))
    assert len(test_db.get_runs()) == 2

    log.info("DELETE test")
    test_db.delete_run("test")
    assert len(test_db.get_runs()) == 1

    log.info("ZIP test")
    test_db.zip_results()
    assert os.path.isfile("results.zip")

    log.info("CLEAR DB test")
    test_db.clear_db()
    assert len(test_db.get_runs()) == 0
    log.info("Finished")
