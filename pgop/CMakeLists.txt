 # Defaults to site-packages so only need package name
 install(DIRECTORY ./ DESTINATION pgop FILES_MATCHING PATTERN "*.py")
 install(FILES ./data.h5 sphere-codes.npz DESTINATION pgop)
