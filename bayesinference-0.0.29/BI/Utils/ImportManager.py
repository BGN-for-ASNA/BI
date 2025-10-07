import concurrent.futures
import importlib

class LazyImporter:
    """
    Manages the background importing of heavy libraries using a thread pool.
    """
    def __init__(self):
        # Using a ThreadPoolExecutor to run synchronous import statements in the background
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self._futures = {}

    def schedule_import(self, module_name, alias):
        """Schedules a module to be imported in the background."""
        if alias not in self._futures:
            # importlib.import_module is the programmatic equivalent of 'import module_name'
            future = self._executor.submit(importlib.import_module, module_name)
            self._futures[alias] = future

    def get_module(self, alias):
        """

        Gets the imported module. If not yet loaded, it will wait for the import to complete.
        This is the point where the main thread will block if the import isn't finished.
        """
        if alias not in self._futures:
            raise ImportError(f"Module for alias '{alias}' was not scheduled for import.")
        
        # future.result() waits for the background task to complete and returns its result
        return self._futures[alias].result()

    def shutdown(self):
        """Cleans up the executor."""
        self._executor.shutdown()

